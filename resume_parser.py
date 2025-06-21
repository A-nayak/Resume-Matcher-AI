import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import subprocess
import streamlit as st

# --- NLTK Data Setup ---
nltk_data_dir = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        st.success("NLTK 'punkt' tokenizer found.")
    except LookupError:
        st.warning("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        st.success("NLTK 'punkt' downloaded.")

    try:
        nltk.data.find('corpora/stopwords')
        st.success("NLTK 'stopwords' corpus found.")
    except LookupError:
        st.warning("NLTK 'stopwords' corpus not found. Downloading...")
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        st.success("NLTK 'stopwords' downloaded.")

download_nltk_data()
# --- End NLTK Data Setup ---


# --- SpaCy Model Setup ---
_nlp_model = None

@st.cache_resource
def get_spacy_model():
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            st.success("SpaCy model 'en_core_web_sm' loaded successfully!")
        except OSError:
            st.warning("SpaCy model 'en_core_web_sm' not found. Attempting to download. This may take a moment...")
            result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                                    capture_output=True, text=True, check=True)
            if result.returncode == 0:
                st.success("SpaCy model 'en_core_web_sm' downloaded successfully!")
            else:
                st.error(f"Failed to download spaCy model: {result.stderr}")
                raise RuntimeError("Failed to download spaCy model.")
            _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model
# --- End SpaCy Model Setup ---


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def extract_info_spacy(text):
    nlp_model = get_spacy_model()
    doc = nlp_model(text)
    extracted_data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": []
    }

    email_match = re.search(r'\S+@\S+', text)
    if email_match: extracted_data["email"] = email_match.group(0)

    phone_match = re.search(r'\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b', text)
    if phone_match: extracted_data["phone"] = phone_match.group(0)

    common_skills = [
        "python", "java", "c++", "javascript", "html", "css", "react", "angular", "vue",
        "machine learning", "deep learning", "nlp", "artificial intelligence",
        "data analysis", "sql", "nosql", "aws", "azure", "google cloud", "docker", "kubernetes",
        "git", "agile", "scrum", "project management", "microsoft office", "excel", "powerpoint",
        "communication", "teamwork", "leadership", "problem-solving"
    ]
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            extracted_data["skills"].append(skill.capitalize())

    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            ent_text_lower = ent.text.lower()
            if "university" in ent_text_lower or "college" in ent_text_lower or "institute" in ent_text_lower:
                extracted_data["education"].append(ent.text)
            elif "company" in ent_text_lower or "inc" in ent_text_lower or "llc" in ent_text_lower or "corp" in ent_text_lower or "group" in ent_text_lower:
                extracted_data["experience"].append(ent.text)

    if not extracted_data["name"] and doc.ents:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                extracted_data["name"] = ent.text
                break
    if not extracted_data["name"]:
        extracted_data["name"] = "Name Not Extracted (Placeholder)"

    extracted_data["skills"] = list(set(extracted_data["skills"]))
    extracted_data["education"] = list(set(extracted_data["education"]))
    extracted_data["experience"] = list(set(extracted_data["experience"]))

    return extracted_data

if __name__ == "__main__":
    st.write("Running resume_parser.py directly for testing:")
    sample_resume_text = """
    John Doe
    john.doe@example.com
    (123) 456-7890
    Software Engineer with 5 years of experience in Python and Java. Strong background in Machine Learning and NLP.
    Education: Master of Science in Computer Science from Stanford University (2020)
    Experience: Lead Software Engineer at Google Inc. (2020-Present), Software Developer at Microsoft Corp. (2018-2020)
    Skills: Python, Java, C++, Machine Learning, Natural Language Processing, Data Analysis
    """

    st.write("--- Testing clean_text ---")
    cleaned_text = clean_text(sample_resume_text)
    st.write(f"Cleaned Text: {cleaned_text}")

    st.write("--- Testing extract_info_spacy ---")
    extracted_data = extract_info_spacy(cleaned_text)
    st.json(extracted_data)

    st.write("Ensuring spaCy model loads:")
    get_spacy_model()
    st.write("Ensuring NLTK data is checked:")
    download_nltk_data()
