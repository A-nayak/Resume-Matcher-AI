import nltk # Rake-NLTK internally uses NLTK
from rake_nltk import Rake
# Import the function to get the spaCy model from resume_parser
from resume_parser import get_spacy_model

def extract_keywords_rake(text):
    # Ensure NLTK punkt and stopwords are downloaded (handled in resume_parser.py)
    # Rake-NLTK needs them, and if resume_parser.py is imported first, they'll be there.
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_keywords_spacy(text):
    # Get the spaCy model using the robust getter from resume_parser
    nlp_model = get_spacy_model()
    doc = nlp_model(text)
    # Extract entities that are likely skills or relevant concepts
    keywords = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "LOC", "PERSON", "NORP"]]
    # You can refine this to include specific POS tags as well, e.g., nouns, adjectives
    # Example: Add nouns and proper nouns that are not already entities
    # keywords.extend([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and token.text not in keywords])
    return list(set(keywords))
