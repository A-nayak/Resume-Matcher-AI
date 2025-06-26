import os
import magic
from PyPDF2 import PdfReader
from docx import Document
import pdfminer.high_level
from typing import Optional

class TextExtractor:
    def __init__(self):
        self.mime = magic.Magic(mime=True)

    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from PDF, DOCX, or TXT files"""
        try:
            file_type = self.mime.from_file(file_path)

            if file_type == "application/pdf":
                return self._extract_from_pdf(file_path)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._extract_from_docx(file_path)
            elif file_type == "text/plain":
                return self._extract_from_txt(file_path)
            return None
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None

    def _extract_from_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from PDF using PyPDF2 (fallback to pdfminer)"""
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if text.strip():
                    return text
        except:
            return pdfminer.high_level.extract_text(file_path)

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
