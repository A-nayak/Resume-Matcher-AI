from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MatchingEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        tfidf = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
