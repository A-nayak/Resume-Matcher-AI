import logging

from rake_nltk import Rake
import nltk

logger = logging.getLogger(__name__)

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Downloading NLTK punkt & stopwords...")
    nltk.download("punkt")
    nltk.download("stopwords")

def extract_keywords_rake(text: str, max_phrases: int = 20) -> list[str]:
    """
    Extract top-keyphrases using RAKE.
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()[:max_phrases]
    return phrases

def extract_keywords_spacy(text: str, nlp) -> list[str]:
    """
    Extract named-entities as keywords via spaCy.
    """
    doc = nlp(text)
    keywords = {
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"ORG", "GPE", "PRODUCT", "LOC", "PERSON"}
    }
    return list(keywords)
