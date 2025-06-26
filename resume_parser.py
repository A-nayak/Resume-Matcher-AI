import re
import os
import magic
import pandas as pd
from typing import Dict, Optional, List
from docx import Document
from PyPDF2 import PdfReader
import pdfminer.high_level
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

class ResumeParser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.skill_patterns = [
            {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["python", "java", "c++"]}}]},
            {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "analysis"}]}
        ]
        self.ruler = nlp.add_pipe("entity_ruler", before="ner")
        self.ruler.add_patterns(self.skill_patterns)

    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from file based on its type"""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)

            if file_type == "application/pdf":
                # Try PyPDF2 first
                try:
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        if text.strip():
                            return text
                except:
                    # Fallback to pdfminer
                    return pdfminer.high_level.extract_text(file_path)
            
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            elif file_type == "text/plain":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return None
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s.,;:!?]', '', text)  # Remove special chars
        return text.strip()

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using spaCy NER and pattern matching"""
        doc = nlp(text.lower())
        skills = set()
        
        # Extract using NER
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skills.add(ent.text)
        
        # Additional keyword matching
        skill_keywords = ["python", "java", "c++", "machine learning", "data analysis", 
                         "sql", "javascript", "html", "css", "react", "angular", "node.js",
                         "tensorflow", "pytorch", "flask", "django", "aws", "azure"]
        
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token in skill_keywords and token not in self.stop_words:
                skills.add(token)
        
        return list(skills)

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience"""
        experience = []
        # Simple regex pattern for dates and positions
        pattern = r'(?P<position>[A-Z][a-zA-Z\s]+)\s*(?:at|in|,)\s*(?P<company>[A-Z][a-zA-Z\s]+)\s*(?P<duration>\(.*?\))'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            experience.append({
                "position": match.group("position").strip(),
                "company": match.group("company").strip(),
                "duration": match.group("duration").strip("()")
            })
        
        return experience

    def parse_resume(self, file_path: str) -> Dict:
        """Main parsing function"""
        text = self.extract_text(file_path)
        if not text:
            return {}
        
        clean_text = self.clean_text(text)
        
        return {
            "skills": self.extract_skills(clean_text),
            "experience": self.extract_experience(clean_text),
            "raw_text": clean_text
        }
