import streamlit as st
import tempfile
import os
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

def main():
    st.set_page_config(page_title="Resume Matcher AI", layout="wide")
    st.title("📄 Resume Matcher AI")
    
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"])
    
    if st.button("Analyze") and resume_file and jd_file:
        with st.spinner("Processing..."):
            # Save files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
                tmp_resume.write(resume_file.getvalue())
                resume_path = tmp_resume.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
                tmp_jd.write(jd_file.getvalue())
                jd_path = tmp_jd.name
            
            # Process files
            resume_text = extract_text(resume_path)
            jd_text = extract_text(jd_path)
            
            if not resume_text or not jd_text:
                st.error("Failed to extract text from files")
                return
            
            # Calculate match
            score = calculate_similarity(resume_text, jd_text)
            
            # Display results
            st.success(f"Match Score: {score:.2f}%")
            st.progress(score/100)
            
            # Cleanup
            os.unlink(resume_path)
            os.unlink(jd_path)

if __name__ == "__main__":
    main()
