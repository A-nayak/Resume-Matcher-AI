import re
import spacy
from PyPDF2 import PdfReader
from docx import Document
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))

    def extract_text(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return " ".join([para.text for para in doc.paragraphs])
        else:
            with open(file_path, 'r') as f:
                return f.read()

    def parse_resume(self, file_path: str) -> dict:
        text = self.extract_text(file_path)
        doc = self.nlp(text)
        
        return {
            "skills": list({ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"}),
            "experience": re.findall(r'(?i)(?:worked at|experience|at)\s+([A-Z][a-zA-Z0-9\s&]+)', text),
            "education": re.findall(r'(?i)(?:university|college|degree)[\s:]*([A-Z][a-zA-Z\s]+)', text),
            "raw_text": text
        }
