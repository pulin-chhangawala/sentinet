"""
predict.py - Classify sentiment of new text

Usage:
    python predict.py --model models/ --text "This movie was absolutely fantastic!"
    python predict.py --model models/ --file reviews.txt
"""

import argparse
import pickle
import sys

from preprocess import preprocess, clean_text
from embeddings import document_vector


def load_model(model_dir):
    """Load trained classifier and feature model."""
    with open(f'{model_dir}/classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    method = model_data['method']
    clf = model_data['classifier']
    vector_size = model_data.get('vector_size', 100)
    
    if method == 'word2vec':
        from gensim.models import Word2Vec
        w2v = Word2Vec.load(f'{model_dir}/word2vec.model')
        return clf, w2v, method, vector_size
    else:
        with open(f'{model_dir}/tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return clf, tfidf, method, vector_size


def predict_text(text, clf, feature_model, method, vector_size=100):
    """Predict sentiment for a single text."""
    if method == 'word2vec':
        tokens = preprocess(text)
        vec = document_vector(tokens, feature_model, vector_size).reshape(1, -1)
    else:
        cleaned = clean_text(text)
        vec = feature_model.transform([cleaned])
    
    prob = clf.predict_proba(vec)[0]
    label = "POSITIVE" if prob[1] >= 0.5 else "NEGATIVE"
    confidence = max(prob)
    
    return label, confidence, prob[1]


def main():
    parser = argparse.ArgumentParser(description='Predict sentiment')
    parser.add_argument('--model', default='models/', help='Model directory')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', '-t', nargs='+', help='Text to classify')
    group.add_argument('--file', '-f', help='File with one text per line')
    
    args = parser.parse_args()
    
    clf, feature_model, method, vector_size = load_model(args.model)
    
    if args.text:
        texts = [' '.join(args.text)]
    else:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    
    print(f"{'Sentiment':<12} {'Conf':<8} {'P(pos)':<8} Text")
    print("-" * 70)
    
    for text in texts:
        label, conf, pos_prob = predict_text(
            text, clf, feature_model, method, vector_size)
        snippet = text[:50] + "..." if len(text) > 50 else text
        print(f"{label:<12} {conf:<8.3f} {pos_prob:<8.3f} {snippet}")


if __name__ == '__main__':
    main()
