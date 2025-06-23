import logging
from pathlib import Path

import PyPDF2
import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> str:
    text = []
    try:
        with pdf_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
    except Exception as e:
        logger.exception(f"Failed to extract PDF text: {e}")
        raise
    return "\n".join(text)

def extract_text_from_docx(docx_path: Path) -> str:
    text = []
    try:
        doc = docx.Document(docx_path)
        text = [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        logger.exception(f"Failed to extract DOCX text: {e}")
        raise
    return "\n".join(text)

def extract_text_from_txt(txt_path: Path) -> str:
    try:
        return txt_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.exception(f"Failed to read TXT file: {e}")
        raise

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt")
