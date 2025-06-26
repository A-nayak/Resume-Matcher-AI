import re
import spacy
from PyPDF2 import PdfReader
from docx import Document
import nltk
from nltk.corpus import stopwords
from typing import Dict, List, Optional

nltk.download('punkt')
nltk.download('stopwords')

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))

    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from PDF/DOCX files."""
        if file_path.endswith('.pdf'):
            return self._extract_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._extract_from_docx(file_path)
        return None

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2."""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def parse_resume(self, file_path: str) -> Dict:
        """Parse resume and extract skills/experience."""
        text = self.extract_text(file_path)
        if not text:
            return {}

        doc = self.nlp(text)
        skills = list({ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"})
        
        return {
            "skills": skills,
            "experience": self._extract_experience(text),
            "education": self._extract_education(text),
            "raw_text": text
        }

    def _extract_experience(self, text: str) -> List[str]:
        """Extract work experience (simplified)."""
        return re.findall(r'(?i)(?:worked at|experience|at)\s+([A-Z][a-zA-Z0-9\s&]+)', text)

    def _extract_education(self, text: str) -> List[str]:
        """Extract education (simplified)."""
        return re.findall(r'(?i)(?:university|college|degree|bachelor|master|phd)[\s:]*([A-Z][a-zA-Z\s]+)', text)
