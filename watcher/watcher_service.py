import os
import time
import json
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    project_root: Path
    watcher_root: Path

    inbox_dir: Path
    out_invoice: Path
    out_email: Path
    out_paper: Path
    out_unsure: Path

    log_dir: Path
    log_file: Path

    text_model_path: Path
    vision_model_dir: Path

    confidence_threshold: float = 0.90

    # PDF text extraction quality threshold:
    # if extracted text < this many chars => treat as "not parsable" and fallback to vision
    min_pdf_text_chars: int = 250

    # file-ready settings (avoid partial writes)
    stable_checks: int = 3
    stable_sleep_s: float = 0.4

    # allow these file types
    allowed_exts: Tuple[str, ...] = (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # Vision inference PDF rendering
    pdf_dpi: int = 200
    vision_max_pages: int = 2

    # Poppler bin path for Windows (used by pdf2image). Set via env var POPPLER_BIN.
    poppler_bin: Optional[str] = None


def default_config() -> Config:
    """
    Expected folder structure:

    document-classifier/
      models/
        text_model/...
        vision_model/...
      watcher/
        watcher_service.py
        inbox/
        invoices/
        e_mails/
        papers/
        unsure/
        logs/
    """
    # watcher_service.py is in .../document-classifier/watcher/watcher_service.py
    watcher_root = Path(__file__).resolve().parent            # .../document-classifier/watcher
    project_root = watcher_root.parent                        # .../document-classifier

    models_dir = project_root / "models"
    log_dir = watcher_root / "logs"

    return Config(
        project_root=project_root,
        watcher_root=watcher_root,

        inbox_dir=watcher_root / "inbox",
        out_invoice=watcher_root / "invoices",
        out_email=watcher_root / "e_mails",
        out_paper=watcher_root / "papers",
        out_unsure=watcher_root / "unsure",

        log_dir=log_dir,
        log_file=log_dir / "watcher_log.jsonl",

        text_model_path=models_dir / "text_model" / "v1" / "tfidf_logreg_baseline.pkl",
        vision_model_dir=models_dir / "vision_model",

        confidence_threshold=float(os.getenv("CONF_THRESH", "0.90")),
        min_pdf_text_chars=int(os.getenv("MIN_PDF_TEXT_CHARS", "250")),

        pdf_dpi=int(os.getenv("PDF_DPI", "200")),
        vision_max_pages=int(os.getenv("VISION_MAX_PAGES", "2")),

        # IMPORTANT: do NOT hardcode user-specific paths
        poppler_bin=os.getenv("POPPLER_BIN"),
    )


# ----------------------------
# Utilities
# ----------------------------

def sha256_12(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def ensure_dirs(cfg: Config) -> None:
    for d in [cfg.inbox_dir, cfg.out_invoice, cfg.out_email, cfg.out_paper, cfg.out_unsure, cfg.log_dir]:
        d.mkdir(parents=True, exist_ok=True)


def append_log(cfg: Config, entry: Dict[str, Any]) -> None:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    with cfg.log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def is_file_ready(path: Path, checks: int, sleep_s: float) -> bool:
    """
    Returns True if file size is stable across N checks.
    Helps avoid reading a file while it is still being copied into inbox.
    """
    try:
        last = path.stat().st_size
    except FileNotFoundError:
        return False

    stable = 0
    for _ in range(checks):
        time.sleep(sleep_s)
        try:
            cur = path.stat().st_size
        except FileNotFoundError:
            return False

        if cur == last and cur > 0:
            stable += 1
        else:
            stable = 0
        last = cur

    return stable >= (checks - 1)


def safe_move(src: Path, dst_dir: Path) -> Path:
    """
    Move src into dst_dir. If name exists, add a suffix.
    Returns final path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    stem, suf = src.stem, src.suffix
    for i in range(1, 9999):
        cand = dst_dir / f"{stem}__{i}{suf}"
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand

    raise RuntimeError(f"Could not move {src}; too many duplicates in {dst_dir}")


# ----------------------------
# Text extraction
# ----------------------------

def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: Path, max_pages: int = 2) -> str:
    """
    Extract text from first N pages using pypdf.
    Works well for born-digital PDFs with an embedded text layer.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    texts = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        page = reader.pages[i]
        t = page.extract_text() or ""
        texts.append(t)

    text = "\n".join(texts)
    text = " ".join(text.split())
    return text.strip()


# ----------------------------
# Vision helpers (PDF->images, image loading)
# ----------------------------

def pdf_to_pil_images(cfg: Config, pdf_path: Path) -> List["PIL.Image.Image"]:
    """
    Convert first cfg.vision_max_pages pages of a PDF to PIL images using pdf2image.
    On Windows, requires POPPLER_BIN to be set to the Poppler /bin path.
    """
    from pdf2image import convert_from_path

    kwargs = dict(
        dpi=cfg.pdf_dpi,
        first_page=1,
        last_page=cfg.vision_max_pages,
    )

    # Only pass poppler_path if provided; keeps it portable across OS
    if cfg.poppler_bin:
        kwargs["poppler_path"] = cfg.poppler_bin

    pages = convert_from_path(str(pdf_path), **kwargs)

    # ensure RGB
    pages = [p.convert("RGB") for p in pages]
    return pages


def load_image_as_pil(img_path: Path) -> "PIL.Image.Image":
    from PIL import Image
    return Image.open(img_path).convert("RGB")


# ----------------------------
# Models
# ----------------------------

class Predictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.text_model = None

        # vision components lazy-loaded
        self._vision_loaded = False
        self._vision_processor = None
        self._vision_model = None
        self._vision_device = None

        self._load_text_model()

    def _load_text_model(self):
        if not self.cfg.text_model_path.exists():
            raise FileNotFoundError(f"Text model not found: {self.cfg.text_model_path}")
        self.text_model = joblib.load(self.cfg.text_model_path)

    def _load_vision_model_if_needed(self):
        if self._vision_loaded:
            return
        if not self.cfg.vision_model_dir.exists():
            raise FileNotFoundError(f"Vision model dir not found: {self.cfg.vision_model_dir}")

        import json
        import torch
        from transformers import AutoModelForImageClassification

        model_dir = self.cfg.vision_model_dir

        # 1) Load model
        self._vision_model = AutoModelForImageClassification.from_pretrained(model_dir)
        self._vision_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vision_model.to(self._vision_device).eval()

        # 2) Try to load processor in a robust way
        processor = None

        try:
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception as e1:
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_dir)
            except Exception as e2:
                cfg_path = Path(model_dir) / "config.json"
                if not cfg_path.exists():
                    raise RuntimeError(
                        f"Vision model config.json not found at {cfg_path}. "
                        f"Cannot infer image processor. Original errors: {repr(e1)} / {repr(e2)}"
                    )

                cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
                model_type = cfg_json.get("model_type", None)

                mapping = {
                    "vit": ("transformers", "ViTImageProcessor"),
                    "deit": ("transformers", "DeiTImageProcessor"),
                    "beit": ("transformers", "BeitImageProcessor"),
                    "convnext": ("transformers", "ConvNextImageProcessor"),
                    # NOTE: resnet mapping may differ; keep as fallback only
                    "resnet": ("transformers", "ConvNextImageProcessor"),
                    "swin": ("transformers", "ViTImageProcessor"),
                }

                if model_type in mapping:
                    module_name, class_name = mapping[model_type]
                    mod = __import__(module_name, fromlist=[class_name])
                    cls = getattr(mod, class_name)
                    processor = cls.from_pretrained(model_dir)
                else:
                    raise RuntimeError(
                        "Could not load image processor.\n"
                        f"- AutoImageProcessor error: {repr(e1)}\n"
                        f"- AutoProcessor error: {repr(e2)}\n"
                        f"- config.json model_type = {model_type!r}\n\n"
                        "Fix options:\n"
                        "1) Upgrade transformers:  pip install -U transformers\n"
                        "2) Ensure your vision_model folder includes preprocessor_config.json with key "
                        "`image_processor_type`\n"
                        "3) Tell me your model_type and I’ll add the correct processor mapping."
                    )

        self._vision_processor = processor
        self._vision_loaded = True

    def predict_with_text_model(self, text: str) -> Tuple[str, Dict[str, float], float]:
        """
        Returns (label, prob_dict, confidence)
        """
        proba = self.text_model.predict_proba([text])[0]
        classes = list(getattr(self.text_model, "classes_", []))
        pred_idx = int(np.argmax(proba))
        label = str(classes[pred_idx]) if classes else str(self.text_model.predict([text])[0])

        prob_dict = {str(c): float(p) for c, p in zip(classes, proba)}
        conf = float(np.max(proba)) if len(proba) else 0.0
        return label, prob_dict, conf

    def predict_with_vision_images(self, images: List["PIL.Image.Image"]) -> Tuple[str, Dict[str, float], float]:
        """
        Run image classifier page-wise and aggregate by taking the page with highest confidence.
        Returns (label, prob_dict_for_best_page, confidence_best_page)
        """
        self._load_vision_model_if_needed()

        import torch

        results = []
        id2label = self._vision_model.config.id2label

        for page_idx, img in enumerate(images, start=1):
            inputs = self._vision_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._vision_device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._vision_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            pred_id = int(torch.argmax(probs).item())
            conf = float(probs[pred_id].item())

            label = id2label.get(str(pred_id), id2label[pred_id]) if isinstance(id2label, dict) else id2label[pred_id]
            prob_dict = {
                (id2label.get(str(i), id2label[i]) if isinstance(id2label, dict) else id2label[i]): float(probs[i].item())
                for i in range(len(probs))
            }

            results.append((page_idx, str(label), conf, prob_dict))

        best = max(results, key=lambda x: x[2])
        _, best_label, best_conf, best_probs = best
        return best_label, best_probs, float(best_conf)


# ----------------------------
# Routing / label mapping
# ----------------------------

def route_folder(cfg: Config, label: str, confidence: float) -> Path:
    if confidence < cfg.confidence_threshold:
        return cfg.out_unsure

    normalized = label.upper().strip()

    mapping = {
        "INVOICE": cfg.out_invoice,
        "EMAIL": cfg.out_email,
        "E_MAIL": cfg.out_email,
        "SCIENTIFIC_PAPER": cfg.out_paper,
        "PAPER": cfg.out_paper,
    }

    return mapping.get(normalized, cfg.out_unsure)


# ----------------------------
# Watcher handler
# ----------------------------

class InboxHandler(FileSystemEventHandler):
    def __init__(self, cfg: Config, predictor: Predictor):
        self.cfg = cfg
        self.predictor = predictor

    def on_created(self, event):
        if event.is_directory:
            return
        self._process(Path(event.src_path))

    def on_moved(self, event):
        if event.is_directory:
            return
        self._process(Path(event.dest_path))

    def _process(self, path: Path):
        try:
            try:
                path = path.resolve()
            except Exception:
                path = Path(str(path))

            if path.suffix.lower() not in self.cfg.allowed_exts:
                return
            if self.cfg.inbox_dir.resolve() not in path.parents:
                return

            if not is_file_ready(path, checks=self.cfg.stable_checks, sleep_s=self.cfg.stable_sleep_s):
                time.sleep(1.0)
                if not is_file_ready(path, checks=self.cfg.stable_checks, sleep_s=self.cfg.stable_sleep_s):
                    append_log(self.cfg, {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file": str(path.name),
                        "hash": None,
                        "status": "skipped_not_ready",
                        "error": None
                    })
                    return

            fhash = sha256_12(path)

            final_model = None  # "text" or "vision"
            used_text_model = False
            used_vision_model = False

            label = "UNSURE"
            confidence = 0.0
            probs: Dict[str, float] = {}

            ext = path.suffix.lower()

            # ---------- TXT: always text model ----------
            if ext == ".txt":
                text = extract_text_from_txt(path)
                if text.strip():
                    used_text_model = True
                    label, probs, confidence = self.predictor.predict_with_text_model(text)
                    final_model = "text"
                else:
                    label, probs, confidence = "UNSURE", {}, 0.0

            # ---------- PDF: try text extraction first; escalate to vision if low confidence ----------
            elif ext == ".pdf":
                text = extract_text_from_pdf(path, max_pages=2)

                if len(text) >= self.cfg.min_pdf_text_chars:
                    used_text_model = True
                    label, probs, confidence = self.predictor.predict_with_text_model(text)
                    final_model = "text"

                    if confidence < self.cfg.confidence_threshold:
                        used_vision_model = True
                        images = pdf_to_pil_images(self.cfg, path)
                        v_label, v_probs, v_conf = self.predictor.predict_with_vision_images(images)

                        if v_conf >= self.cfg.confidence_threshold or v_conf > confidence:
                            label, probs, confidence = v_label, v_probs, v_conf
                            final_model = "vision"
                else:
                    used_vision_model = True
                    images = pdf_to_pil_images(self.cfg, path)
                    label, probs, confidence = self.predictor.predict_with_vision_images(images)
                    final_model = "vision"

                    if confidence < self.cfg.confidence_threshold and text.strip():
                        used_text_model = True
                        t_label, t_probs, t_conf = self.predictor.predict_with_text_model(text)

                        if t_conf >= self.cfg.confidence_threshold or t_conf > confidence:
                            label, probs, confidence = t_label, t_probs, t_conf
                            final_model = "text"

            # ---------- Images: direct vision model ----------
            else:
                used_vision_model = True
                img = load_image_as_pil(path)
                label, probs, confidence = self.predictor.predict_with_vision_images([img])
                final_model = "vision"

            target_dir = route_folder(self.cfg, label, confidence)
            moved_to = safe_move(path, target_dir)

            append_log(self.cfg, {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": str(path.name),
                "hash": fhash,
                "text_classifier_sufficient": "yes" if final_model == "text" else "no",
                "vision_fallback_used": "yes" if used_vision_model else "no",
                "predicted_class": str(label),
                "confidence": float(confidence),
                "moved_to": str(moved_to.parent.name),
                "status": "moved",
                "error": None,
            })

        except Exception as e:
            append_log(self.cfg, {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": str(getattr(path, "name", "unknown")),
                "hash": None,
                "status": "error",
                "error": repr(e)
            })


def main():
    cfg = default_config()
    ensure_dirs(cfg)

    print("=== Watcher config ===")
    print("Project root:", cfg.project_root)
    print("Watcher root:", cfg.watcher_root)
    print("Inbox:", cfg.inbox_dir)
    print("Outputs:", cfg.out_invoice, cfg.out_email, cfg.out_paper, cfg.out_unsure)
    print("Text model:", cfg.text_model_path)
    print("Vision model dir:", cfg.vision_model_dir)
    print("Confidence threshold:", cfg.confidence_threshold)
    print("Min PDF text chars:", cfg.min_pdf_text_chars)
    print("PDF DPI:", cfg.pdf_dpi, "Max pages:", cfg.vision_max_pages)
    print("Poppler bin:", cfg.poppler_bin)
    print("Log:", cfg.log_file)
    print("======================")

    predictor = Predictor(cfg)

    handler = InboxHandler(cfg, predictor)
    observer = Observer()
    observer.schedule(handler, str(cfg.inbox_dir), recursive=False)
    observer.start()

    print("Watcher running. Drop files into watcher/inbox/. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping watcher...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
