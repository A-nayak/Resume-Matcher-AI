import nltk
from rake_nltk import Rake
from resume_parser import get_spacy_model

def extract_keywords_rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_keywords_spacy(text):
    nlp_model = get_spacy_model()
    doc = nlp_model(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "LOC", "PERSON", "NORP"]]
    return list(set(keywords))
