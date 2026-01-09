"""
DistilBERT Classifier for Guna Classification
==============================================
5-fold stratified CV with fine-tuning

Compares against TF-IDF + LogReg baseline.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 8,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'n_folds': 5,
    'num_labels': 3,
    'random_seed': 42
}

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# =============================================================================
# DATASET CLASS
# =============================================================================

class GunaDataset(Dataset):
    """PyTorch Dataset for Guna classification."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> tuple[list[str], np.ndarray]:
    """Load text and labels from CSV."""
    df = pd.read_csv(filepath)
    texts = df['text'].tolist()
    labels = df['label'].values
    return texts, labels


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions."""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(texts: list[str], labels: np.ndarray, config: dict) -> dict:
    """Run stratified k-fold cross-validation with DistilBERT."""

    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_seed'])

    fold_results = []
    all_preds = np.zeros_like(labels)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n--- Fold {fold}/{config['n_folds']} ---")

        # Split data
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Create datasets
        train_dataset = GunaDataset(train_texts, train_labels, tokenizer, config['max_length'])
        test_dataset = GunaDataset(test_texts, test_labels, tokenizer, config['max_length'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

        # Initialize model
        model = DistilBertForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        model.to(DEVICE)

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
        total_steps = len(train_loader) * config['epochs']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(config['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            print(f"  Epoch {epoch+1}: Loss = {train_loss:.4f}")

        # Evaluate
        preds, _ = evaluate(model, test_loader, DEVICE)
        all_preds[test_idx] = preds

        # Calculate metrics
        f1_macro = f1_score(test_labels, preds, average='macro')
        f1_weighted = f1_score(test_labels, preds, average='weighted')

        fold_results.append({
            'fold': fold,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'test_size': len(test_idx)
        })

        print(f"  F1 (macro) = {f1_macro:.4f}, F1 (weighted) = {f1_weighted:.4f}")

        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'fold_results': fold_results,
        'all_preds': all_preds,
        'true_labels': labels
    }


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_final_metrics(results: dict) -> dict:
    """Compute overall metrics from cross-validation results."""

    fold_results = results['fold_results']
    true_labels = results['true_labels']
    all_preds = results['all_preds']

    # Aggregate fold metrics
    f1_macros = [r['f1_macro'] for r in fold_results]
    f1_weighted_scores = [r['f1_weighted'] for r in fold_results]

    # Overall metrics
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


def save_results(metrics: dict, filepath: str, config: dict) -> None:
    """Save results to text file."""

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DISTILBERT CLASSIFIER\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['learning_rate']}\n")
        f.write(f"Max length: {config['max_length']}\n")
        f.write(f"Validation: {config['n_folds']}-fold Stratified Cross-Validation\n\n")

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

        # Comparison with baseline
        f.write("\n" + "=" * 60 + "\n")
        f.write("COMPARISON WITH BASELINE\n")
        f.write("=" * 60 + "\n")
        f.write("Baseline (TF-IDF + LogReg): Macro F1 = 0.3500 ± 0.0790\n")
        f.write(f"DistilBERT:                 Macro F1 = {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}\n")
        diff = metrics['f1_macro_mean'] - 0.35
        f.write(f"Difference:                 {diff:+.4f}\n")

    print(f"\nResults saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""

    # Paths
    DATA_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/data/guna_classifier_data.csv'
    RESULTS_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/results/distilbert_results.txt'

    print("\n" + "#" * 60)
    print("  DISTILBERT CLASSIFIER FOR GUNA CLASSIFICATION")
    print("#" * 60)

    print(f"\nDevice: {DEVICE}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")

    # Load data
    print("\nLoading data...")
    texts, labels = load_data(DATA_PATH)
    print(f"Loaded {len(texts)} samples")
    print(f"Class distribution: Tamas={sum(labels==0)}, Rajas={sum(labels==1)}, Sattva={sum(labels==2)}")

    # Run cross-validation
    results = run_cross_validation(texts, labels, CONFIG)

    # Compute final metrics
    metrics = compute_final_metrics(results)

    # Print results
    print_results(metrics)

    # Save results
    save_results(metrics, RESULTS_PATH, CONFIG)

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    print(f"Baseline (TF-IDF + LogReg): Macro F1 = 0.3500 ± 0.0790")
    print(f"DistilBERT:                 Macro F1 = {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}")
    diff = metrics['f1_macro_mean'] - 0.35
    if diff > 0.05:
        print(f"Improvement: +{diff:.4f} ✅ Deep learning helps!")
    elif diff > 0:
        print(f"Improvement: +{diff:.4f} (marginal)")
    else:
        print(f"Difference: {diff:.4f} ❌ No improvement")

    print("\n" + "#" * 60)
    print("  COMPLETE")
    print("#" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    metrics = main()
