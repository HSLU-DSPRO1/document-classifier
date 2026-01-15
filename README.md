# Document Classifier for DSPRO1

This project classifies incoming documents into predefined classes (e.g. `INVOICE`, `EMAIL`, `SCIENTIFIC_PAPER`).
It uses:

- **Text classifier** (fast): for `.txt` and text-based PDFs (extractable text layer)
- **Vision classifier** (fallback): for scanned PDFs and images (`.png/.jpg/...`)
- **Watcher service**: monitors an inbox folder and automatically routes files into output folders

---
<br>

## How to use

1. Install requirements

```
pip install -r requirements.txt

```

2. Start the watcher service

```
python watcher/watcher_service.py

```

3. Drop files into watcher/inbox/

---

<br>

## Windows Note: Poppler (required for scanned PDFs)

If you want to classify scanned PDFs on Windows, `pdf2image` requires **Poppler**.

### Option A: Temporary (current terminal session)

1. Install Poppler (Windows)
2. Set the environment variable in the terminal, then run the watcher script
PowerShell:

```powershell
$env:POPPLER_BIN="C:\path\to\poppler\Library\bin"
python watcher\watcher_service.py
```

### Option B: Permanent (system environment variable)

1. Install Poppler (Windows)
2. Add a new environment variable:
   - **Name:** `POPPLER_BIN`
   - **Value:** `C:\path\to\poppler\Library\bin`
3. Restart the terminal and run the watcher normally.

---

<br>
<br>
<br>
Authors: <br>
Diego Gonzalez <br>
Viacheslav Godovskii