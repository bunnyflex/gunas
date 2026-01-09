"""
Baseline Model: TF-IDF + Logistic Regression
============================================
5-fold stratified CV for Guna classification

Provides baseline performance to compare against DistilBERT.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: Load Data
# =============================================================================

def load_data(filepath: str) -> tuple[list[str], np.ndarray]:
    """Load text and labels from CSV."""
    df = pd.read_csv(filepath)
    texts = df['text'].tolist()
    labels = df['label'].values
    return texts, labels


# =============================================================================
# STEP 2: TF-IDF + Logistic Regression Pipeline
# =============================================================================

def create_tfidf_vectorizer() -> TfidfVectorizer:
    """Create TF-IDF vectorizer with good defaults for short text."""
    return TfidfVectorizer(
        max_features=1000,      # Limit features for small dataset
        ngram_range=(1, 2),     # Unigrams and bigrams
        min_df=2,               # Ignore very rare terms
        max_df=0.95,            # Ignore very common terms
        stop_words='english'
    )


def create_classifier() -> LogisticRegression:
    """Create Logistic Regression with class balancing."""
    return LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,
        random_state=42,
        C=1.0
    )


# =============================================================================
# STEP 3: Cross-Validation
# =============================================================================

def run_cross_validation(texts: list[str], labels: np.ndarray, n_folds: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation.

    Returns:
        dict with fold results, predictions, and metrics
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_preds = np.zeros_like(labels)
    all_probs = np.zeros((len(labels), 3))

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
        # Split data
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Vectorize
        vectorizer = create_tfidf_vectorizer()
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Train
        clf = create_classifier()
        clf.fit(X_train, train_labels)

        # Predict
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        # Store predictions
        all_preds[test_idx] = preds
        all_probs[test_idx] = probs

        # Calculate metrics
        f1_macro = f1_score(test_labels, preds, average='macro')
        f1_weighted = f1_score(test_labels, preds, average='weighted')

        fold_results.append({
            'fold': fold,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'test_size': len(test_idx)
        })

        print(f"\nFold {fold}: F1 (macro) = {f1_macro:.4f}, F1 (weighted) = {f1_weighted:.4f}")

    return {
        'fold_results': fold_results,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'true_labels': labels
    }


# =============================================================================
# STEP 4: Evaluation Metrics
# =============================================================================

def compute_final_metrics(results: dict) -> dict:
    """Compute overall metrics from cross-validation results."""

    fold_results = results['fold_results']
    true_labels = results['true_labels']
    all_preds = results['all_preds']

    # Aggregate fold metrics
    f1_macros = [r['f1_macro'] for r in fold_results]
    f1_weighted_scores = [r['f1_weighted'] for r in fold_results]

    # Overall metrics on all predictions
    overall_f1_macro = f1_score(true_labels, all_preds, average='macro')
    overall_f1_weighted = f1_score(true_labels, all_preds, average='weighted')

    # Per-class F1
    f1_per_class = f1_score(true_labels, all_preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(true_labels, all_preds)

    return {
        'f1_macro_mean': np.mean(f1_macros),
        'f1_macro_std': np.std(f1_macros),
        'f1_weighted_mean': np.mean(f1_weighted_scores),
        'f1_weighted_std': np.std(f1_weighted_scores),
        'overall_f1_macro': overall_f1_macro,
        'overall_f1_weighted': overall_f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            true_labels, all_preds,
            target_names=['Tamas', 'Rajas', 'Sattva']
        )
    }


def print_results(metrics: dict) -> None:
    """Pretty print final results."""

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"\nMacro F1:    {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted_mean']:.4f} ± {metrics['f1_weighted_std']:.4f}")

    print("\n" + "-" * 60)
    print("PER-CLASS F1 SCORES")
    print("-" * 60)
    labels = ['Tamas (0)', 'Rajas (1)', 'Sattva (2)']
    for label, f1 in zip(labels, metrics['f1_per_class']):
        print(f"  {label}: {f1:.4f}")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print("              Pred_T  Pred_R  Pred_S")
    cm = metrics['confusion_matrix']
    row_labels = ['Actual_T', 'Actual_R', 'Actual_S']
    for label, row in zip(row_labels, cm):
        print(f"  {label}    {row[0]:4d}    {row[1]:4d}    {row[2]:4d}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(metrics['classification_report'])


def save_results(metrics: dict, filepath: str) -> None:
    """Save results to text file."""

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BASELINE MODEL: TF-IDF + LOGISTIC REGRESSION\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write("Vectorizer: TF-IDF (max_features=1000, ngram_range=(1,2))\n")
        f.write("Classifier: Logistic Regression (class_weight='balanced')\n")
        f.write("Validation: 5-fold Stratified Cross-Validation\n\n")

        f.write("RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Macro F1:    {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted_mean']:.4f} ± {metrics['f1_weighted_std']:.4f}\n\n")

        f.write("PER-CLASS F1\n")
        f.write("-" * 60 + "\n")
        labels = ['Tamas (0)', 'Rajas (1)', 'Sattva (2)']
        for label, f1 in zip(labels, metrics['f1_per_class']):
            f.write(f"  {label}: {f1:.4f}\n")

        f.write("\nCONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write("              Pred_T  Pred_R  Pred_S\n")
        cm = metrics['confusion_matrix']
        row_labels = ['Actual_T', 'Actual_R', 'Actual_S']
        for label, row in zip(row_labels, cm):
            f.write(f"  {label}    {row[0]:4d}    {row[1]:4d}    {row[2]:4d}\n")

        f.write("\nCLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(metrics['classification_report'])

    print(f"\nResults saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""

    # Configuration
    DATA_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/data/guna_classifier_data.csv'
    RESULTS_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/results/baseline_results.txt'
    N_FOLDS = 5

    print("\n" + "#" * 60)
    print("  BASELINE MODEL: TF-IDF + LOGISTIC REGRESSION")
    print("#" * 60)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data(DATA_PATH)
    print(f"Loaded {len(texts)} samples")
    print(f"Class distribution: Tamas={sum(labels==0)}, Rajas={sum(labels==1)}, Sattva={sum(labels==2)}")

    # Run cross-validation
    results = run_cross_validation(texts, labels, n_folds=N_FOLDS)

    # Compute final metrics
    metrics = compute_final_metrics(results)

    # Print results
    print_results(metrics)

    # Save results
    save_results(metrics, RESULTS_PATH)

    print("\n" + "#" * 60)
    print("  BASELINE COMPLETE")
    print("#" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    metrics = main()
