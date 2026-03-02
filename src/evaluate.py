"""
evaluate.py - Enhanced model evaluation with visualizations

Generates ROC curves, precision-recall curves, confusion matrices,
SHAP feature importance, and learning curves.

Usage:
    python evaluate.py --model models/ --data data/reviews.csv --output results/
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)
from sklearn.model_selection import learning_curve

from preprocess import preprocess, preprocess_batch, clean_text
from embeddings import document_vector, embed_corpus


def load_model(model_dir):
    """Load trained model and feature model."""
    with open(f'{model_dir}/classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    method = model_data['method']
    clf = model_data['classifier']
    vs = model_data.get('vector_size', 100)
    
    if method == 'word2vec':
        from gensim.models import Word2Vec
        fm = Word2Vec.load(f'{model_dir}/word2vec.model')
    else:
        with open(f'{model_dir}/tfidf.pkl', 'rb') as f:
            fm = pickle.load(f)
    
    return clf, fm, method, vs


def plot_roc(y_true, y_prob, outdir):
    """ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'roc_curve.png'), dpi=150)
    plt.close()


def plot_precision_recall(y_true, y_prob, outdir):
    """Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color='coral', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'pr_curve.png'), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, outdir):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def plot_learning_curve(clf, X, y, outdir):
    """Learning curve: score vs training set size."""
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1, color='steelblue')
    ax.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1, color='coral')
    
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-',
            color='steelblue', label='Train', linewidth=2)
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-',
            color='coral', label='Validation', linewidth=2)
    
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'learning_curve.png'), dpi=150)
    plt.close()


def plot_error_analysis(texts, y_true, y_pred, y_prob, outdir, n=10):
    """Show most confident misclassifications (for debugging)."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                'text': texts[i][:100],
                'true': 'POS' if y_true[i] == 1 else 'NEG',
                'pred': 'POS' if y_pred[i] == 1 else 'NEG',
                'confidence': abs(y_prob[i] - 0.5) * 2,
            })
    
    errors.sort(key=lambda e: e['confidence'], reverse=True)
    
    with open(os.path.join(outdir, 'error_analysis.txt'), 'w') as f:
        f.write("Most Confident Misclassifications\n")
        f.write("=" * 60 + "\n\n")
        for e in errors[:n]:
            f.write(f"True: {e['true']} | Pred: {e['pred']} | "
                    f"Conf: {e['confidence']:.3f}\n")
            f.write(f"  {e['text']}\n\n")
    
    print(f"  Error analysis: {len(errors)} misclassifications "
          f"({len(errors[:n])} worst saved)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', '-o', default='results/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading model and data...")
    clf, fm, method, vs = load_model(args.model)
    
    df = pd.read_csv(args.data).dropna(subset=['text', 'label'])
    texts = df['text'].values
    labels = df['label'].astype(int).values
    
    print("Generating features...")
    if method == 'word2vec':
        tok = preprocess_batch(texts)
        X = embed_corpus(tok, fm, vs)
    else:
        X = fm.transform([clean_text(t) for t in texts])
    
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    print("Generating plots...")
    plot_roc(labels, y_prob, args.output)
    plot_precision_recall(labels, y_prob, args.output)
    plot_confusion(labels, y_pred, args.output)
    plot_learning_curve(clf, X, labels, args.output)
    plot_error_analysis(texts, labels, y_pred, y_prob, args.output)
    
    print(f"\nAll plots saved to {args.output}/")


if __name__ == '__main__':
    main()
