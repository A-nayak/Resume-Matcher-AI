from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict

class MatchingEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return np.round(similarity[0][0] * 100, 2)
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return 0.0

    def match_resume_to_jd(self, resume_data: Dict, jd_text: str) -> Dict:
        """Comprehensive matching between resume and job description"""
        # Text similarity
        similarity_score = self.calculate_similarity(resume_data['raw_text'], jd_text)
        
        return {
            "similarity_score": similarity_score,
            "resume_skills": resume_data.get('skills', []),
            "resume_education": resume_data.get('education', []),
            "resume_experience": resume_data.get('experience', [])
        }
