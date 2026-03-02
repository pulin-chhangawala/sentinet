# sentinet

Text sentiment classifier using Word2Vec embeddings and logistic regression. Built on NLTK for preprocessing and Gensim for word embeddings. Compares Word2Vec document vectors against a TF-IDF baseline.

Honestly started this because I listed NLTK and Word2Vec on my resume and realized I didn't have a project that actually used them beyond import statements.

## Quick Start

```bash
pip install -r requirements.txt

# generate sample data (or use your own CSV with text + label columns)
python data/generate_data.py data/reviews.csv 4000

# train with Word2Vec embeddings
python src/train.py --data data/reviews.csv --output models/

# predict
python src/predict.py --model models/ -t "This movie was absolutely incredible"

# compare: train with TF-IDF baseline
python src/train.py --data data/reviews.csv --use-tfidf --output models_tfidf/
```

## How It Works

### Pipeline
1. **Preprocessing** (NLTK): HTML stripping → lowercasing → contraction expansion → tokenization → stopword removal (preserving negation words!) → lemmatization
2. **Feature extraction**: Either Word2Vec document vectors (avg pooling) or TF-IDF
3. **Classification**: Logistic regression with L2 regularization
4. **Evaluation**: Accuracy, F1, AUC, 5-fold cross-validation

### Why keep negation words?

Most stopword lists remove "not", "no", "never", etc. But "this movie was not good" has literally opposite sentiment from "this movie was good". Removing "not" destroys the signal. This is the kind of thing you only discover when you look at misclassified examples.

### Why Word2Vec + Logistic Regression (not BERT)?

For learning and interpretability. Word2Vec lets you inspect exactly what the model captures about word relationships:

```python
# after training:
model.wv.most_similar("terrible")
# → ['dreadful', 'awful', 'horrible', ...]

model.wv.most_similar("brilliant")  
# → ['superb', 'outstanding', 'magnificent', ...]
```

BERT would get better accuracy, but you lose the ability to understand *why*.

## Project Structure

```
src/
├── preprocess.py    # Text cleaning, tokenization, lemmatization
├── embeddings.py    # Word2Vec training + document vector computation
├── train.py         # Full training pipeline (W2V or TF-IDF → LR)
└── predict.py       # CLI predictor for new text
data/
└── generate_data.py # Synthetic review generation for testing
```

## Data Format

CSV with columns:
| Column | Description |
|--------|-------------|
| `text` | Review text |
| `label` | 1 = positive, 0 = negative |

## Requirements

- Python 3.8+
- NLTK, Gensim, Scikit-learn, Pandas, NumPy
