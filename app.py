import streamlit as st
from text_extractor import extract_text
# Removed 'nlp as spacy_nlp' as it's no longer exported and not needed directly here
from resume_parser import extract_info_spacy, clean_text, get_spacy_model
from matching_engine import get_embedding, calculate_similarity
# skill_suggester.py's extract_keywords_spacy no longer needs nlp_model passed
from skill_suggester import extract_keywords_rake, extract_keywords_spacy

st.title("AI Resume Analyzer + Smart Internship Matcher")

st.header("Upload Resume")
resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if resume_file:
    st.write("Resume uploaded successfully!")
    # Save the uploaded file temporarily
    with open(f"./{resume_file.name}", "wb") as f:
        f.write(resume_file.getbuffer())

    # Extract text
    resume_text = extract_text(resume_file.name)
    st.subheader("Extracted Resume Text:")
    st.write(resume_text)

    # Clean and parse resume
    cleaned_resume_text = clean_text(resume_text)
    parsed_resume_data = extract_info_spacy(cleaned_resume_text)
    st.subheader("Parsed Resume Data:")
    st.json(parsed_resume_data)

    st.header("Enter Job Description")
    job_description = st.text_area("Paste the job description here:", height=200)

    if job_description:
        st.subheader("Job Description:")
        st.write(job_description)

        # Clean job description (using the same clean_text function)
        cleaned_job_description = clean_text(job_description)

        # Get embeddings and calculate similarity
        resume_embedding = get_embedding(cleaned_resume_text)
        job_description_embedding = get_embedding(cleaned_job_description)
        similarity_score = calculate_similarity(resume_embedding, job_description_embedding)

        st.subheader("Matching Score:")
        st.write(f"The similarity between your resume and the job description is: {similarity_score:.2f}")

        # Basic Recommendation (for now, just based on similarity score)
        if similarity_score > 0.7:
            st.success("Great Match! This internship seems highly relevant to your profile.")
        elif similarity_score > 0.5:
            st.info("Good Match! This internship is relevant, consider applying.")
        else:
            st.warning("Moderate Match. You might need to tailor your resume more for this role.")

        st.header("Skill Suggestions")
        st.write("Based on the job description, here are some suggested skills:")

        # Rake-NLTK based keyword extraction
        rake_keywords = extract_keywords_rake(job_description)
        st.subheader("Keywords (Rake-NLTK):")
        st.write(rake_keywords)

        # spaCy based keyword extraction - no longer passing spacy_nlp
        spacy_keywords = extract_keywords_spacy(job_description)
        st.subheader("Keywords (spaCy):")
        st.write(spacy_keywords)

    # Clean up temporary file (uncomment if you want to enable this)
    # import os
    # os.remove(resume_file.name)
