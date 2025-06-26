import streamlit as st
import tempfile
import os
from resume_parser import ResumeParser
from skill_suggester import SkillSuggester
from matching_engine import MatchingEngine
import pandas as pd

# Set up the app
st.set_page_config(
    page_title="Resume Matcher AI",
    page_icon="📄",
    layout="wide"
)

# Initialize components
resume_parser = ResumeParser()
skill_suggester = SkillSuggester()
matching_engine = MatchingEngine()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    st.title("📄 Resume Matcher AI")
    st.markdown("Upload your resume and job description to analyze the match")
    
    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader("Choose resume file", type=["pdf", "docx", "txt"], key="resume")
    
    with col2:
        st.subheader("Upload Job Description")
        jd_file = st.file_uploader("Choose job description file", type=["pdf", "docx", "txt"], key="jd")
    
    if st.button("Analyze Match") and resume_file and jd_file:
        with st.spinner("Processing files..."):
            # Save files temporarily
            resume_path = save_uploaded_file(resume_file)
            jd_path = save_uploaded_file(jd_file)
            
            if resume_path and jd_path:
                try:
                    # Parse files
                    resume_data = resume_parser.parse_resume(resume_path)
                    jd_text = resume_parser.text_extractor.extract_text(jd_path)
                    
                    if not jd_text:
                        st.error("Could not extract text from job description")
                        return
                    
                    # Perform matching
                    match_result = matching_engine.match_resume_to_jd(resume_data, jd_text)
                    suggestions = skill_suggester.suggest_skills(resume_data, jd_text)
                    
                    # Display results
                    st.success("Analysis complete!")
                    
                    # Overall match score
                    st.subheader(f"Overall Match Score: {match_result['similarity_score']}%")
                    st.progress(match_result['similarity_score'] / 100)
                    
                    # Skills analysis
                    st.subheader("Skills Analysis")
                    
                    tab1, tab2, tab3 = st.tabs(["Your Skills", "Missing Skills", "Recommendations"])
                    
                    with tab1:
                        if suggestions['current_skills']:
                            st.write(pd.DataFrame({"Your Skills": suggestions['current_skills']}))
                        else:
                            st.warning("No skills detected in resume")
                    
                    with tab2:
                        if suggestions['missing_skills']:
                            st.write(pd.DataFrame({"Missing Skills": suggestions['missing_skills']}))
                        else:
                            st.success("All key skills are present!")
                    
                    with tab3:
                        if suggestions['recommendations']:
                            for skill, recs in suggestions['recommendations'].items():
                                st.markdown(f"**For {skill}:**")
                                st.write(", ".join(recs))
                        else:
                            st.info("No specific recommendations needed")
                    
                    # Detailed view
                    with st.expander("View Detailed Analysis"):
                        st.subheader("Resume Education")
                        st.write(resume_data.get('education', []))
                        
                        st.subheader("Resume Experience")
                        st.write(resume_data.get('experience', []))
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Clean up temporary files
                    if os.path.exists(resume_path):
                        os.unlink(resume_path)
                    if os.path.exists(jd_path):
                        os.unlink(jd_path)

if __name__ == "__main__":
    main()
