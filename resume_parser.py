import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import Dict, List
from text_extractor import TextExtractor

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')

class ResumeParser:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.stop_words = set(stopwords.words('english'))
        self.skill_patterns = self._get_skill_patterns()
        self.ruler = nlp.add_pipe("entity_ruler", before="ner")
        self.ruler.add_patterns(self.skill_patterns)

    def _get_skill_patterns(self) -> List[Dict]:
        """Define patterns for skill extraction"""
        return [
            {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["python", "java", "c++"]}}]},
            {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "science"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "web"}, {"LOWER": "development"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "artificial"}, {"LOWER": "intelligence"}]}
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?]', '', text)
        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        doc = nlp(text)
        entities = {
            "skills": set(),
            "education": set(),
            "experience": set(),
            "organizations": set()
        }
        
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                entities["skills"].add(ent.text)
            elif ent.label_ in ["ORG", "COMPANY"]:
                entities["organizations"].add(ent.text)
            elif ent.label_ == "EDU":
                entities["education"].add(ent.text)
        
        return {k: list(v) for k, v in entities.items()}

    def parse_resume(self, file_path: str) -> Dict:
        """Parse resume and extract structured information"""
        text = self.text_extractor.extract_text(file_path)
        if not text:
            return {}
        
        clean_text = self.clean_text(text)
        entities = self.extract_entities(clean_text)
        
        return {
            **entities,
            "raw_text": clean_text,
            "file_path": file_path
        }
