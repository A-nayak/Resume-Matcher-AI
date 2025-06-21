import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def extract_info_spacy(text):
    doc = nlp(text)
    extracted_data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": []
    }

    # Extracting Name, Email, Phone (basic regex for now)
    # This is a simplified approach, more robust regex or NER training would be needed for production
    email_match = re.search(r'\S+@\S+', text)
    if email_match: extracted_data["email"] = email_match.group(0)

    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    if phone_match: extracted_data["phone"] = phone_match.group(0)

    # Extracting skills (simplified - will need a more comprehensive approach)
    # For a real system, this would involve a predefined list of skills or custom NER
    keywords = ["python", "java", "c++", "machine learning", "nlp", "data analysis", "project management"]
    for keyword in keywords:
        if keyword in text:
            extracted_data["skills"].append(keyword)

    # Extracting Education and Experience (using rule-based approach with spaCy entities)
    # This is a very basic example and would need significant refinement for real-world resumes
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "GPE": # Organizations or Geopolitical Entities might be part of education/experience
            if "university" in ent.text.lower() or "college" in ent.text.lower():
                extracted_data["education"].append(ent.text)
            elif "company" in ent.text.lower() or "inc" in ent.text.lower() or "llc" in ent.text.lower():
                extracted_data["experience"].append(ent.text)

    # A more sophisticated approach for name extraction would involve NER models trained on names
    # For now, a placeholder
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

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def extract_info_spacy(text):
    doc = nlp(text)
    extracted_data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": []
    }

    # Extracting Name, Email, Phone (basic regex for now)
    # This is a simplified approach, more robust regex or NER training would be needed for production
    email_match = re.search(r'\S+@\S+', text)
    if email_match: extracted_data["email"] = email_match.group(0)

    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    if phone_match: extracted_data["phone"] = phone_match.group(0)

    # Extracting skills (simplified - will need a more comprehensive approach)
    # For a real system, this would involve a predefined list of skills or custom NER
    keywords = ["python", "java", "c++", "machine learning", "nlp", "data analysis", "project management"]
    for keyword in keywords:
        if keyword in text:
            extracted_data["skills"].append(keyword)

    # Extracting Education and Experience (using rule-based approach with spaCy entities)
    # This is a very basic example and would need significant refinement for real-world resumes
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "GPE": # Organizations or Geopolitical Entities might be part of education/experience
            if "university" in ent.text.lower() or "college" in ent.text.lower():
                extracted_data["education"].append(ent.text)
            elif "company" in ent.text.lower() or "inc" in ent.text.lower() or "llc" in ent.text.lower():
                extracted_data["experience"].append(ent.text)

    # A more sophisticated approach for name extraction would involve NER models trained on names
    # For now, a placeholder
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

