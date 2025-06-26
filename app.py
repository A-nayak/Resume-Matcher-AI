import streamlit as st
import tempfile
import os
from resume_parser import ResumeParser
from matching_engine import MatchingEngine

st.set_page_config(page_title="Resume Matcher AI", layout="wide")

def main():
    st.title("📄 Resume Matcher AI")
    st.write("Upload a resume and job description to check compatibility.")

    # File upload
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])

    if st.button("Analyze Match") and resume_file and jd_file:
        with st.spinner("Processing..."):
            # Save files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
                tmp_resume.write(resume_file.read())
                resume_path = tmp_resume.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
                tmp_jd.write(jd_file.read())
                jd_path = tmp_jd.name

            # Parse files
            parser = ResumeParser()
            resume_data = parser.parse_resume(resume_path)
            jd_text = parser.extract_text(jd_path)

            # Calculate match
            matcher = MatchingEngine()
            similarity = matcher.calculate_similarity(resume_data["raw_text"], jd_text)
            
            # Display results
            st.success(f"Match Score: {similarity:.2f}%")
            st.progress(similarity / 100)

            st.subheader("Resume Skills")
            st.write(resume_data["skills"])

            # Cleanup
            os.unlink(resume_path)
            os.unlink(jd_path)

if __name__ == "__main__":
    main()
