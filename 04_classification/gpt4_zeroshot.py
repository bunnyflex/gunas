"""
GPT-4 Zero-Shot Guna Classification
====================================
No training - uses GPT-4's understanding of Vedic psychology.

Compares against baseline and DistilBERT.
"""

import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Load .env file
load_dotenv('/Users/sunny/Downloads/IOS/gunas/.env')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'model': 'gpt-4',  # or 'gpt-4-turbo' for faster/cheaper
    'temperature': 0,   # Deterministic output
    'max_tokens': 10,   # Only need one word
    'sleep_between_calls': 0.5  # Rate limiting
}

# System prompt with Guna definitions
SYSTEM_PROMPT = """You are an expert in Vedic psychology and the Triguna framework.

Your task is to classify text into ONE of three Gunas based on the psychological state expressed:

SATTVA: Clarity, balance, self-awareness, calm insight, acceptance, wisdom, harmony, mindfulness, genuine understanding, peaceful resolve

RAJAS: Striving, desire, agitation, ambition, restlessness, active effort, motivation, seeking change, energy in motion, driven action

TAMAS: Inertia, confusion, denial, hopelessness, avoidance, stuck, resistance, darkness, overwhelm, inability to act, cognitive fog

Focus on the underlying psychological state, not just surface emotions."""

USER_PROMPT_TEMPLATE = """Classify the following text into ONE Guna.

Text: "{text}"

Respond with ONLY one word: Sattva, Rajas, or Tamas"""


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
# GPT-4 CLASSIFICATION
# =============================================================================

def classify_single(client: OpenAI, text: str, config: dict) -> str:
    """Classify a single text using GPT-4."""

    response = client.chat.completions.create(
        model=config['model'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ]
    )

    return response.choices[0].message.content.strip()


def parse_response(response: str) -> int:
    """Parse GPT-4 response to label (0=Tamas, 1=Rajas, 2=Sattva)."""

    response_lower = response.lower()

    if 'sattva' in response_lower:
        return 2
    elif 'rajas' in response_lower:
        return 1
    elif 'tamas' in response_lower:
        return 0
    else:
        # If unclear, return -1 (will handle as error)
        return -1


def classify_all(texts: list[str], config: dict) -> tuple[np.ndarray, list[str]]:
    """Classify all texts using GPT-4."""

    # Initialize client
    client = OpenAI()  # Uses OPENAI_API_KEY env variable

    predictions = []
    raw_responses = []
    errors = 0

    print("\n" + "=" * 60)
    print("CLASSIFYING WITH GPT-4")
    print("=" * 60)

    for i, text in enumerate(texts):
        try:
            # Get response
            response = classify_single(client, text, config)
            raw_responses.append(response)

            # Parse to label
            label = parse_response(response)
            if label == -1:
                print(f"  [{i+1}] Warning: Unclear response '{response}'")
                errors += 1
                label = 0  # Default to Tamas if unclear

            predictions.append(label)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(texts)} sentences...")

            # Rate limiting
            time.sleep(config['sleep_between_calls'])

        except Exception as e:
            print(f"  [{i+1}] Error: {str(e)}")
            predictions.append(0)  # Default
            raw_responses.append(f"ERROR: {str(e)}")
            errors += 1

    print(f"\nCompleted: {len(texts)} sentences, {errors} errors")

    return np.array(predictions), raw_responses


# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute classification metrics."""

    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_per_class = f1_score(true_labels, predictions, average=None)
    cm = confusion_matrix(true_labels, predictions)

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            true_labels, predictions,
            target_names=['Tamas', 'Rajas', 'Sattva']
        )
    }


def print_results(metrics: dict) -> None:
    """Pretty print results."""

    print("\n" + "=" * 60)
    print("GPT-4 ZERO-SHOT RESULTS")
    print("=" * 60)

    print(f"\nMacro F1:    {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

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


def save_results(metrics: dict, predictions: np.ndarray, raw_responses: list[str],
                 texts: list[str], true_labels: np.ndarray, filepath: str, config: dict) -> None:
    """Save results to text file."""

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GPT-4 ZERO-SHOT CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Temperature: {config['temperature']}\n")
        f.write(f"Method: Zero-shot with Guna definitions in prompt\n\n")

        f.write("RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Macro F1:    {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n\n")

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

        # Comparison
        f.write("\n" + "=" * 60 + "\n")
        f.write("COMPARISON WITH OTHER MODELS\n")
        f.write("=" * 60 + "\n")
        f.write("Baseline (TF-IDF + LogReg): Macro F1 = 0.3500\n")
        f.write(f"GPT-4 Zero-Shot:            Macro F1 = {metrics['f1_macro']:.4f}\n")
        diff = metrics['f1_macro'] - 0.35
        f.write(f"Difference vs Baseline:     {diff:+.4f}\n")

    print(f"\nResults saved to: {filepath}")

    # Also save detailed predictions
    detail_path = filepath.replace('.txt', '_detailed.csv')
    detail_df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predictions,
        'gpt4_response': raw_responses,
        'correct': true_labels == predictions
    })
    detail_df.to_csv(detail_path, index=False)
    print(f"Detailed predictions saved to: {detail_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""

    # Paths
    DATA_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/data/guna_classifier_data.csv'
    RESULTS_PATH = '/Users/sunny/Downloads/IOS/gunas/04_classification/results/gpt4_results.txt'

    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n" + "!" * 60)
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("!" * 60)
        print("\nSet it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nThen run this script again.")
        return None

    print("\n" + "#" * 60)
    print("  GPT-4 ZERO-SHOT GUNA CLASSIFICATION")
    print("#" * 60)

    print(f"\nModel: {CONFIG['model']}")
    print(f"Temperature: {CONFIG['temperature']}")

    # Load data
    print("\nLoading data...")
    texts, labels = load_data(DATA_PATH)
    print(f"Loaded {len(texts)} samples")
    print(f"Class distribution: Tamas={sum(labels==0)}, Rajas={sum(labels==1)}, Sattva={sum(labels==2)}")

    # Classify
    predictions, raw_responses = classify_all(texts, CONFIG)

    # Compute metrics
    metrics = compute_metrics(labels, predictions)

    # Print results
    print_results(metrics)

    # Save results
    save_results(metrics, predictions, raw_responses, texts, labels, RESULTS_PATH, CONFIG)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    print(f"Baseline (TF-IDF + LogReg): Macro F1 = 0.3500")
    print(f"GPT-4 Zero-Shot:            Macro F1 = {metrics['f1_macro']:.4f}")
    diff = metrics['f1_macro'] - 0.35
    if diff > 0.1:
        print(f"Improvement: +{diff:.4f} ✅ GPT-4 significantly better!")
    elif diff > 0.05:
        print(f"Improvement: +{diff:.4f} ✅ GPT-4 helps!")
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
