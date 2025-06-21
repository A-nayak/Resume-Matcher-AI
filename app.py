import streamlit as st
import os
from text_extractor import extract_text
from resume_parser import extract_info_spacy, clean_text, get_spacy_model
from matching_engine import get_embedding, calculate_similarity
from skill_suggester import extract_keywords_rake, extract_keywords_spacy

st.set_page_config(layout="wide", page_title="AI Resume Matcher")

st.title("🤖 AI Resume Analyzer + Smart Internship Matcher")
st.markdown("Upload your resume and paste a job description to find out how well you match and get skill suggestions!")

st.header("Upload Your Resume 📄")
resume_file = st.file_uploader("Choose a resume file (PDF or DOCX)", type=["pdf", "docx"])

resume_text = None
if resume_file:
    st.success("Resume uploaded successfully! Processing...")
    file_path = os.path.join("/tmp", resume_file.name)
    with open(file_path, "wb") as f:
        f.write(resume_file.getbuffer())

    try:
        resume_text = extract_text(file_path)
        st.subheader("Extracted Resume Text Preview:")
        st.expander("Click to view full resume text").write(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

        cleaned_resume_text = clean_text(resume_text)
        parsed_resume_data = extract_info_spacy(cleaned_resume_text)
        st.subheader("Parsed Resume Data:")
        st.json(parsed_resume_data)

    except Exception as e:
        st.error(f"Error processing resume: {e}")
        st.info("Please ensure your resume is a valid PDF or DOCX file and not corrupted.")

    finally:
        os.remove(file_path)

st.header("Enter Job Description 📋")
job_description = st.text_area("Paste the job description here:", height=300,
                               placeholder="e.g., We are looking for a Software Engineer with strong Python and Machine Learning skills...")

if job_description and resume_text:
    st.subheader("Job Description Preview:")
    st.expander("Click to view full job description").write(job_description[:1000] + "..." if len(job_description) > 1000 else job_description)

    cleaned_job_description = clean_text(job_description)

    try:
        resume_embedding = get_embedding(cleaned_resume_text)
        job_description_embedding = get_embedding(cleaned_job_description)
        similarity_score = calculate_similarity(resume_embedding, job_description_embedding)

        st.subheader("Matching Score 🎯")
        st.metric(label="Resume-Job Description Similarity", value=f"{similarity_score:.2f}")

        if similarity_score > 0.75:
            st.success("🎉 Excellent Match! Your profile highly aligns with this role. Definitely apply!")
        elif similarity_score > 0.5:
            st.info("👍 Good Match! This role seems relevant. Tailor your resume slightly to improve your chances.")
        else:
            st.warning("🧐 Moderate Match. You might need to significantly tailor your resume or look for other roles where your skills are a better fit.")

        st.header("Skill Suggestions for Job Description ✨")
        st.write("Here are some key skills extracted from the job description:")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Keywords (Rake-NLTK):")
            rake_keywords = extract_keywords_rake(job_description)
            if rake_keywords:
                for keyword in rake_keywords[:10]:
                    st.markdown(f"- {keyword}")
            else:
                st.write("No keywords found using Rake-NLTK.")

        with col2:
            st.subheader("Keywords (spaCy NER):")
            spacy_keywords = extract_keywords_spacy(job_description)
            if spacy_keywords:
                for keyword in spacy_keywords[:10]:
                    st.markdown(f"- {keyword}")
            else:
                st.write("No keywords found using spaCy NER.")

    except Exception as e:
        st.error(f"Error during matching or skill suggestion: {e}")
        st.info("Please ensure job description is sufficient for analysis.")

elif job_description and not resume_text:
    st.info("Please upload a resume to get a matching score and skill suggestions.")
elif resume_text and not job_description:
    st.info("Please paste a job description to get a matching score and skill suggestions.")
