import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import subprocess # Added for spaCy model download

# --- NLTK Data Setup ---
nltk_data_dir = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir # Set the environment variable for NLTK

# Download NLTK data (punkt for tokenization, stopwords for cleaning)
# NLTK will check if the data is already present before downloading.
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
# --- End NLTK Data Setup ---

# --- SpaCy Model Setup ---
_nlp_model = None

def get_spacy_model():
    """Loads and returns the spaCy model, ensuring it's loaded only once."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            # If the model isn't found, download it. This is crucial for deployment.
            st.warning("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True, text=True)
            _nlp_model = spacy.load("en_core_web_sm")
            st.success("SpaCy model downloaded and loaded successfully!")
    return _nlp_model
# --- End SpaCy Model Setup ---

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def extract_info_spacy(text):
    # Get the spaCy model using the getter function
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

    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    if phone_match: extracted_data["phone"] = phone_match.group(0)

    keywords = ["python", "java", "c++", "machine learning", "nlp", "data analysis", "project management"]
    for keyword in keywords:
        if keyword in text:
            extracted_data["skills"].append(keyword)

    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "GPE":
            if "university" in ent.text.lower() or "college" in ent.text.lower():
                extracted_data["education"].append(ent.text)
            elif "company" in ent.text.lower() or "inc" in ent.text.lower() or "llc" in ent.text.lower():
                extracted_data["experience"].append(ent.text)

    extracted_data["name"] = "Extracted Name Placeholder"

    return extracted_data

if __name__ == "__main__":
    sample_resume_text = """
    John Doe
    john.doe@example.com
    (123) 456-7890
    Software Engineer with 5 years of experience in Python and Java. Strong background in Machine Learning and NLP.
    Education: Master of Science in Computer Science from Stanford University (2020)
    Experience: Lead Software Engineer at Google Inc. (2020-Present), Software Developer at Microsoft Corp. (2018-2020)
    Skills: Python, Java, C++, Machine Learning, Natural Language Processing, Data Analysis
    """

    cleaned_text = clean_text(sample_resume_text)
    print("Cleaned Text:", cleaned_text)

    extracted_data = extract_info_spacy(cleaned_text)
    print("Extracted Data:", extracted_data)
