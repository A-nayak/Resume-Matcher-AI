import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-load the embedding model once
_model = None
def _get_model():
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)

def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    score = cosine_similarity([emb1], [emb2])[0][0]
    return float(score)
