
import nltk
from rake_nltk import Rake

def extract_keywords_rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_keywords_spacy(text, nlp_model):
    doc = nlp_model(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "LOC", "PERSON"]]
    # You can refine this to include specific POS tags as well, e.g., nouns, adjectives
    return list(set(keywords))

