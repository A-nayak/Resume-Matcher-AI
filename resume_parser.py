import logging
import re

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Ensure NLTK stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}"
)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    stops = set(stopwords.words("english"))
    return " ".join([t for t in tokens if t not in stops and t.strip()])

def extract_info(text: str) -> dict:
    nlp = load_spacy_model()

    data = {"name": "", "email": "", "phone": "", "skills": [], "education": [], "experience": []}

    data["email"] = EMAIL_REGEX.search(text).group(0) if EMAIL_REGEX.search(text) else ""
    data["phone"] = PHONE_REGEX.search(text).group(0) if PHONE_REGEX.search(text) else ""

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            data["name"] = ent.text
            break

    SKILL_KEYWORDS = {"python", "java", "c++", "machine learning", "nlp", "data analysis", "sql"}
    for token in set(text.split()):
        if token in SKILL_KEYWORDS:
            data["skills"].append(token)

    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(k in sent_text for k in ("university", "college", "institute")):
            data["education"].append(sent.text.strip())
        if any(k in sent_text for k in ("inc", "llc", "corp", "company", "startup")):
            data["experience"].append(sent.text.strip())

    data["skills"] = sorted(set(data["skills"]))
    data["education"] = sorted(set(data["education"]))
    data["experience"] = sorted(set(data["experience"]))
    return data
