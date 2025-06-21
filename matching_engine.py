from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st # Import streamlit for caching

# Load pre-trained Sentence-Transformer model
@st.cache_resource # Use Streamlit's cache to avoid re-loading model on rerun
def get_sentence_transformer_model():
    """Loads the SentenceTransformer model, caching it for efficiency."""
    st.info("Loading Sentence-Transformer model (this happens once)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    st.success("Sentence-Transformer model loaded.")
    return model

# Initialize the model once
model = get_sentence_transformer_model()

def get_embedding(text):
    # Ensure text is not empty, as encode might fail on empty string
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension()) # Return zero vector for empty text
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    # Ensure inputs are not empty vectors if they resulted from empty text
    if np.all(embedding1 == 0) and np.all(embedding2 == 0):
        return 0.0 # Or handle as appropriate if both are empty
    return cosine_similarity([embedding1], [embedding2])[0][0]

if __name__ == "__main__":
    st.write("--- Testing matching_engine.py ---")
    resume_text = "Software Engineer with strong skills in Python, Machine Learning, and NLP. 5 years experience at Google."
    job_description_text = "Looking for a Python Developer with experience in AI and Natural Language Processing. Experience with large datasets."

    st.write("Getting embeddings...")
    resume_embedding = get_embedding(resume_text)
    job_description_embedding = get_embedding(job_description_text)
    st.write("Calculating similarity...")
    similarity_score = calculate_similarity(resume_embedding, job_description_embedding)
    st.write(f"Similarity Score: {similarity_score}")

    st.write("\nTesting with empty text:")
    empty_embedding = get_embedding("")
    similarity_empty = calculate_similarity(resume_embedding, empty_embedding)
    st.write(f"Similarity with empty text: {similarity_empty}")
