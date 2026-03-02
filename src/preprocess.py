"""
preprocess.py - Text preprocessing pipeline for sentiment analysis

Handles tokenization, stopword removal, lemmatization, and text normalization.
Built on NLTK because it gives you way more control than spaCy for
understanding what each step actually does.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download required NLTK data (idempotent)
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words('english'))

# keep negation words because they're critical for sentiment
_keep_words = {'not', 'no', 'nor', 'neither', 'never', 'none',
               'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
               "n't", "not", "don't", "doesn't", "didn't", "won't",
               "wouldn't", "couldn't", "shouldn't", "can't", "isn't",
               "aren't", "wasn't", "weren't", "haven't", "hasn't"}

# trim stopwords but keep negation
_filtered_stops = _stopwords - _keep_words


def clean_text(text):
    """Basic text normalization."""
    # lowercase
    text = text.lower()
    # remove HTML tags (common in review data)
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # handle contractions that NLTK might miss
    text = text.replace("n't", " not")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    # remove punctuation (but keep apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_filter(text):
    """Tokenize, remove stopwords, and lemmatize."""
    tokens = word_tokenize(text)
    
    filtered = []
    for tok in tokens:
        # skip pure punctuation and very short tokens
        if tok in string.punctuation or len(tok) < 2:
            continue
        # skip stopwords (but keep negations)
        if tok in _filtered_stops:
            continue
        # lemmatize
        lemma = _lemmatizer.lemmatize(tok)
        filtered.append(lemma)
    
    return filtered


def preprocess(text):
    """Full pipeline: clean → tokenize → filter → lemmatize."""
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    return tokens


def preprocess_batch(texts):
    """Process a list of texts."""
    return [preprocess(t) for t in texts]
