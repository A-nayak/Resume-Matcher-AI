import streamlit as st
from text_extractor import extract_text
from resume_parser import extract_info_spacy, clean_text, nlp as spacy_nlp
from matching_engine import get_embedding, calculate_similarity
from skill_suggester import extract_keywords_rake, extract_keywords_spacy
import os

st.title("AI Resume Analyzer + Smart Internship Matcher")

# Initialize NLTK (just in case)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.header("Upload Resume")
resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if resume_file:
    try:
        st.write("Resume uploaded successfully!")
        # Save the uploaded file temporarily
        temp_file_path = f"./{resume_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(resume_file.getbuffer())

        # Extract text
        resume_text = extract_text(temp_file_path)
        st.subheader("Extracted Resume Text:")
        st.text_area("", resume_text, height=200)

        # Clean and parse resume
        cleaned_resume_text = clean_text(resume_text)
        parsed_resume_data = extract_info_spacy(cleaned_resume_text)
        st.subheader("Parsed Resume Data:")
        st.json(parsed_resume_data)

        st.header("Enter Job Description")
        job_description = st.text_area("Paste the job description here:", height=200)

        if job_description:
            st.subheader("Job Description:")
            st.text_area("", job_description, height=200)

            # Clean job description
            cleaned_job_description = clean_text(job_description)

            # Get embeddings and calculate similarity
            resume_embedding = get_embedding(cleaned_resume_text)
            job_description_embedding = get_embedding(cleaned_job_description)
            similarity_score = calculate_similarity(resume_embedding, job_description_embedding)

            st.subheader("Matching Score:")
            st.metric("Similarity Score", f"{similarity_score:.2%}")

            # Recommendation
            if similarity_score > 0.7:
                st.success("Great Match! This internship seems highly relevant to your profile.")
            elif similarity_score > 0.5:
                st.info("Good Match! This internship is relevant, consider applying.")
            else:
                st.warning("Moderate Match. You might need to tailor your resume more for this role.")

            st.header("Skill Suggestions")
            st.write("Based on the job description, here are some suggested skills:")

            # Rake-NLTK based keyword extraction
            st.subheader("Keywords (Rake-NLTK):")
            rake_keywords = extract_keywords_rake(job_description)
            st.write(rake_keywords[:10])  # Show top 10 keywords

            # spaCy based keyword extraction
            st.subheader("Keywords (spaCy):")
            spacy_keywords = extract_keywords_spacy(job_description, spacy_nlp)
            st.write(spacy_keywords[:10])  # Show top 10 keywords

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
