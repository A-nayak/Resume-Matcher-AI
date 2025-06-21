import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load spaCy model outside the functions for efficiency, but don't export directly
# It's better to pass it as an argument or have a getter function if truly needed elsewhere.
_nlp_model = None

def get_spacy_model():
    """Loads and returns the spaCy model, ensuring it's loaded only once."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm") #
        except OSError:
            # If the model isn't found, download it. This is crucial for deployment.
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp_model = spacy.load("en_core_web_sm") #
    return _nlp_model

def clean_text(text): #
    text = text.lower() #
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    tokens = word_tokenize(text) #
    stop_words = set(stopwords.words('english')) #
    filtered_tokens = [word for word in tokens if word not in stop_words] #
    return " ".join(filtered_tokens) #

def extract_info_spacy(text): #
    nlp_model = get_spacy_model() # Get the model here
    doc = nlp_model(text) #
    extracted_data = { #
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": []
    }

    email_match = re.search(r'\S+@\S+', text) #
    if email_match: extracted_data["email"] = email_match.group(0) #

    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) #
    if phone_match: extracted_data["phone"] = phone_match.group(0) #

    keywords = ["python", "java", "c++", "machine learning", "nlp", "data analysis", "project management"] #
    for keyword in keywords: #
        if keyword in text: #
            extracted_data["skills"].append(keyword) #

    for ent in doc.ents: #
        if ent.label_ == "ORG" or ent.label_ == "GPE": # Organizations or Geopolitical Entities might be part of education/experience
            if "university" in ent.text.lower() or "college" in ent.text.lower(): #
                extracted_data["education"].append(ent.text) #
            elif "company" in ent.text.lower() or "inc" in ent.text.lower() or "llc" in ent.text.lower(): #
                extracted_data["experience"].append(ent.text) #

    extracted_data["name"] = "Extracted Name Placeholder" #

    return extracted_data #

# Remove the global nlp variable export from here.
# It will now be handled by get_spacy_model()
