"""
train.py - Train a sentiment classifier using Word2Vec + Logistic Regression

Pipeline:
  1. Load and preprocess text data
  2. Train Word2Vec on the corpus
  3. Embed documents as averaged word vectors
  4. Train a Logistic Regression classifier
  5. Evaluate and save model

Usage:
    python train.py --data data/reviews.csv --output models/
    python train.py --data data/reviews.csv --use-tfidf --output models/
"""

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report)

from preprocess import preprocess, preprocess_batch, clean_text
from embeddings import train_word2vec, embed_corpus


def load_data(path):
    """
    Load review data. Expected columns:
      - text: review text
      - label: 1 = positive, 0 = negative
    """
    df = pd.read_csv(path)
    assert 'text' in df.columns and 'label' in df.columns
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df


def train_w2v_pipeline(texts, labels, vector_size=100):
    """Pipeline using Word2Vec embeddings."""
    print("Preprocessing texts...")
    tokenized = preprocess_batch(texts)
    
    print("Training Word2Vec...")
    w2v_model = train_word2vec(tokenized, vector_size=vector_size)
    
    print("Embedding documents...")
    X = embed_corpus(tokenized, w2v_model, vector_size)
    
    return X, w2v_model


def train_tfidf_pipeline(texts):
    """Pipeline using TF-IDF features (baseline comparison)."""
    print("Building TF-IDF features...")
    cleaned = [clean_text(t) for t in texts]
    
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2,
        max_df=0.95,
    )
    X = tfidf.fit_transform(cleaned)
    return X, tfidf


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classifier')
    parser.add_argument('--data', required=True, help='Path to reviews CSV')
    parser.add_argument('--output', default='models/', help='Output directory')
    parser.add_argument('--use-tfidf', action='store_true',
                        help='Use TF-IDF instead of Word2Vec')
    parser.add_argument('--vector-size', type=int, default=100,
                        help='Word2Vec vector dimensions')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # load data
    print("Loading data...")
    df = load_data(args.data)
    print(f"  {len(df)} reviews loaded")
    print(f"  Class balance: {df['label'].mean():.1%} positive")
    
    texts = df['text'].values
    labels = df['label'].values
    
    # split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels)
    
    # feature extraction
    if args.use_tfidf:
        # tfidf needs to fit on train only
        all_texts = texts  # but w2v trains on all for vocab
        X, feature_model = train_tfidf_pipeline(texts)
        X_train = feature_model.transform([clean_text(t) for t in X_train_text])
        X_test = feature_model.transform([clean_text(t) for t in X_test_text])
        method = "tfidf"
    else:
        tokenized_all = preprocess_batch(texts)
        w2v_model = train_word2vec(tokenized_all, vector_size=args.vector_size)
        
        train_tok = preprocess_batch(X_train_text)
        test_tok = preprocess_batch(X_test_text)
        
        X_train = embed_corpus(train_tok, w2v_model, args.vector_size)
        X_test = embed_corpus(test_tok, w2v_model, args.vector_size)
        feature_model = w2v_model
        method = "word2vec"
    
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # train classifier
    print("Training classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
    )
    clf.fit(X_train, y_train)
    
    # evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  Results ({method}):")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    AUC:      {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")
    
    # cross-validation on full set for a more robust estimate
    print("Cross-validation (5-fold)...")
    if method == "word2vec":
        X_all = embed_corpus(preprocess_batch(texts), feature_model, args.vector_size)
    else:
        X_all = feature_model.transform([clean_text(t) for t in texts])
    
    cv_scores = cross_val_score(clf, X_all, labels, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # save
    model_data = {
        'classifier': clf,
        'method': method,
        'vector_size': args.vector_size,
    }
    
    if method == "word2vec":
        feature_model.save(os.path.join(args.output, 'word2vec.model'))
    else:
        with open(os.path.join(args.output, 'tfidf.pkl'), 'wb') as f:
            pickle.dump(feature_model, f)
    
    with open(os.path.join(args.output, 'classifier.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    meta = {
        'method': method,
        'accuracy': acc, 'f1': f1, 'auc': auc,
        'cv_accuracy': float(cv_scores.mean()),
        'n_train': len(y_train), 'n_test': len(y_test),
    }
    with open(os.path.join(args.output, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Model saved to {args.output}")


if __name__ == '__main__':
    main()
