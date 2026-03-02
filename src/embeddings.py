"""
embeddings.py - Word2Vec embedding features for sentiment classification

Trains a Word2Vec model on the corpus, then represents each document as
the average of its word vectors. Simple but surprisingly effective; this
approach (avg word embeddings) was competitive with RNNs on many
benchmarks before BERT came along.
"""

import numpy as np
from gensim.models import Word2Vec


def train_word2vec(tokenized_corpus, vector_size=100, window=5, 
                   min_count=2, epochs=20):
    """
    Train Word2Vec on the corpus.
    
    Using skip-gram (sg=1) because it handles rare words better than CBOW,
    which matters for sentiment where specific adjectives carry most of
    the signal.
    """
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram
        workers=4,
        epochs=epochs,
        seed=42,
    )
    
    print(f"  Word2Vec trained: {len(model.wv)} words, {vector_size} dimensions")
    return model


def document_vector(tokens, w2v_model, vector_size=100):
    """
    Compute document vector as the mean of word vectors.
    
    Words not in vocabulary are skipped. If no words are in vocab,
    return zero vector.
    """
    vectors = []
    for token in tokens:
        if token in w2v_model.wv:
            vectors.append(w2v_model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    
    return np.mean(vectors, axis=0).astype(np.float32)


def embed_corpus(tokenized_docs, w2v_model, vector_size=100):
    """Compute document vectors for a list of tokenized documents."""
    return np.array([document_vector(doc, w2v_model, vector_size) 
                     for doc in tokenized_docs])


def most_similar_words(w2v_model, word, topn=10):
    """Find words most similar to a given word (useful for debugging)."""
    if word not in w2v_model.wv:
        return []
    return w2v_model.wv.most_similar(word, topn=topn)
