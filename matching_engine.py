from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained Sentence-Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

if __name__ == "__main__":
    resume_text = "Software Engineer with strong skills in Python, Machine Learning, and NLP."
    job_description_text = "Looking for a Python Developer with experience in AI and Natural Language Processing."

    resume_embedding = get_embedding(resume_text)
    job_description_embedding = get_embedding(job_description_text)

    similarity_score = calculate_similarity(resume_embedding, job_description_embedding)
    print(f"Similarity Score: {similarity_score}")

