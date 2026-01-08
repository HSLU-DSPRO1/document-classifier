from huggingface_hub import create_repo, upload_folder
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # basic sanity check
    if not any((folder / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]):
        raise RuntimeError("Model weights not found in folder")

    create_repo(args.repo_id, private=args.private, exist_ok=True)

    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(folder),
        commit_message="Upload trained vision document classifier"
    )

    print(f"✅ Model uploaded to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
