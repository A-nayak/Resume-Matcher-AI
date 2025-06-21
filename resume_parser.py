import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
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

    # Extracting Name, Email, Phone
    email_match = re.search(r'\S+@\S+', text)
    if email_match: extracted_data["email"] = email_match.group(0)

    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    if phone_match: extracted_data["phone"] = phone_match.group(0)

    # Extracting skills
    keywords = ["python", "java", "c++", "machine learning", "nlp", 
               "data analysis", "project management", "sql", "javascript",
               "react", "aws", "docker", "kubernetes"]
    for keyword in keywords:
        if keyword in text:
            extracted_data["skills"].append(keyword)

    # Extracting Education and Experience
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "GPE":
            if "university" in ent.text.lower() or "college" in ent.text.lower():
                extracted_data["education"].append(ent.text)
            elif "company" in ent.text.lower() or "inc" in ent.text.lower() or "llc" in ent.text.lower():
                extracted_data["experience"].append(ent.text)

    # Name extraction (basic)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted_data["name"] = ent.text
            break

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
