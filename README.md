# PDF Query System

Upload a PDF and ask questions about it. The system retrieves the most relevant
passages, answers from those alone, and discards the answer if it drifts too far
from them. It also writes summaries and generates flashcards with a scored quiz.

> Filed as Indian patent application IN 202541123274.
> Applicant: Vellore Institute of Technology. Inventor: Swarnava Banerjee.
> Filed, not granted.


## How it works

### It checks its own answers

Once the model responds, the system scores its answer against the retrieved passages
by word overlap. Below 40 percent it discards the answer and returns "The document
doesn't contain this information." The prompt already tells the model to stay inside
the context, so this checks the output rather than relying on that instruction.

### OCR runs only when it is needed

PyPDF2 extracts the text first. Under 100 words means the file is probably a scan, so
the system renders each page at 300 DPI, converts to greyscale, doubles the contrast,
and runs Tesseract. It keeps that result only if it beats the direct extraction.

## Setup

Python 3.11 or newer. OCR needs two system packages that pip cannot install:

```bash
brew install tesseract poppler                  # macOS
sudo apt install tesseract-ocr poppler-utils    # Debian/Ubuntu
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your OpenAI key, AstraDB token and database ID
streamlit run pdf_system/app.py
```

`SOURCE_DOCUMENTS/` holds two sample PDFs to try it on.

## Known limitations

Only the first 50 chunks of a document get indexed, at 800 characters each, so roughly
its first 40,000 characters. Long files are searchable at the start and invisible after
that. Raising the cap is one line in `pdf_system/utils/vector_store.py`.

Quiz answers are graded by word overlap against the stored answer, and anything over
10 percent counts as correct.

One document at a time. Uploading a new PDF clears the store. No authentication,
since it runs locally.
