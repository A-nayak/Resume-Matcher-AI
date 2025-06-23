import os
import logging
from pathlib import Path

import streamlit as st

from text_extractor import extract_text
from resume_parser import clean_text, extract_info
from resume_parser import load_spacy_model
from matching_engine import get_embedding, calculate_similarity
from skill_suggester import extract_keywords_rake, extract_keywords_spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🚀 AI Resume Analyzer + Smart Internship Matcher")

# Sidebar settings
st.sidebar.header("Settings")
max_rake = st.sidebar.slider("Max RAKE keywords:", min_value=5, max_value=50, value=20)
similarity_threshold = st.sidebar.slider("Good match threshold:", 0.0, 1.0, 0.7)

# File uploader
resume_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if resume_file:
    save_path = Path(resume_file.name)
    with open(save_path, "wb") as f:
        f.write(resume_file.getbuffer())
    st.success("✅ Resume uploaded")

    with st.spinner("Extracting text..."):
        resume_text = extract_text(save_path.as_posix())

    st.subheader("Extracted Text")
    st.write(resume_text)

    # Parsing
    with st.spinner("Parsing resume..."):
        cleaned = clean_text(resume_text)
        info = extract_info(resume_text)

    st.subheader("Parsed Information")
    st.json(info)

    # Download parsed JSON
    st.download_button(
        label="Download JSON",
        data=st.session_state.setdefault("parsed_json", info),
        file_name="parsed_resume.json",
        mime="application/json"
    )

    # Job description input
    st.subheader("🔍 Job Description")
    job_desc = st.text_area("Paste job description:", height=200)

    if job_desc:
        # Similarity
        cleaned_jd = clean_text(job_desc)
        with st.spinner("Calculating similarity..."):
            emb_res = get_embedding(cleaned)
            emb_jd = get_embedding(cleaned_jd)
            score = calculate_similarity(emb_res, emb_jd)

        st.metric("Similarity Score", f"{score:.2f}")
        if score >= similarity_threshold:
            st.success("Great Match! 🎉")
        elif score >= 0.5:
            st.info("Good Match 👍")
        else:
            st.warning("Moderate Match — consider tailoring your resume.")

        # Keyword suggestions
        st.subheader("📋 Skill / Keyword Suggestions")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**RAKE**")
            rake_keys = extract_keywords_rake(job_desc, max_phrases=max_rake)
            st.write(rake_keys)
        with cols[1]:
            st.markdown("**spaCy NER**")
            spacy_keys = extract_keywords_spacy(job_desc, load_spacy_model())
            st.write(spacy_keys)

    # Clean up
    try:
        os.remove(save_path)
    except Exception:
        pass
