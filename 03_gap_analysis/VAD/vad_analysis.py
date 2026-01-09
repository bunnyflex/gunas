"""
VAD (Valence-Arousal-Dominance) Analysis on Guna Annotations
============================================================
Compares NRC-VAD emotional dimensions with Guna classifications
to determine if Gunas capture something beyond VAD.
"""

import pandas as pd
import numpy as np
import re
from scipy import stats

# =============================================================================
# STEP 1: Load NRC-VAD Lexicon
# =============================================================================

def load_vad_lexicon(filepath: str) -> dict:
    """
    Load NRC-VAD lexicon into a dictionary for fast lookup.

    Returns:
        dict: {word: {'valence': v, 'arousal': a, 'dominance': d}}
    """
    df = pd.read_csv(filepath, sep='\t')

    lexicon = {}
    for _, row in df.iterrows():
        # Skip rows with NaN or non-string terms
        if pd.isna(row['term']) or not isinstance(row['term'], str):
            continue
        lexicon[row['term'].lower()] = {
            'valence': row['valence'],
            'arousal': row['arousal'],
            'dominance': row['dominance']
        }

    return lexicon


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, keep only words."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


# =============================================================================
# STEP 2: Compute VAD Scores for Each Sentence
# =============================================================================

def compute_sentence_vad(text: str, lexicon: dict) -> dict:
    """
    Compute average VAD scores for a sentence.

    Returns:
        dict: {'valence': v, 'arousal': a, 'dominance': d, 'matched_words': n}
    """
    words = tokenize(text)

    valence_scores = []
    arousal_scores = []
    dominance_scores = []

    for word in words:
        if word in lexicon:
            valence_scores.append(lexicon[word]['valence'])
            arousal_scores.append(lexicon[word]['arousal'])
            dominance_scores.append(lexicon[word]['dominance'])

    if len(valence_scores) == 0:
        return {
            'valence': np.nan,
            'arousal': np.nan,
            'dominance': np.nan,
            'matched_words': 0,
            'total_words': len(words)
        }

    return {
        'valence': np.mean(valence_scores),
        'arousal': np.mean(arousal_scores),
        'dominance': np.mean(dominance_scores),
        'matched_words': len(valence_scores),
        'total_words': len(words)
    }


# =============================================================================
# STEP 3: Analysis Functions
# =============================================================================

def analyze_vad_by_guna(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean VAD scores for each Guna category.
    """
    summary = results_df.groupby('Guna_Majority').agg({
        'valence': ['mean', 'std'],
        'arousal': ['mean', 'std'],
        'dominance': ['mean', 'std'],
        'Sentence_ID': 'count'
    }).round(3)

    summary.columns = ['V_mean', 'V_std', 'A_mean', 'A_std', 'D_mean', 'D_std', 'count']
    return summary


def compute_correlations(results_df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlation between Guna (numeric) and each VAD dimension.
    Sattva=2, Rajas=1, Tamas=0
    """
    # Remove rows with NaN VAD scores
    df_clean = results_df.dropna(subset=['valence', 'arousal', 'dominance'])

    guna_numeric = df_clean['Guna_Majority'].map({'Sattva': 2, 'Rajas': 1, 'Tamas': 0})

    correlations = {}
    for dim in ['valence', 'arousal', 'dominance']:
        r, p = stats.pearsonr(guna_numeric, df_clean[dim])
        correlations[dim] = {'r': r, 'p': p}

    return correlations


def print_vad_summary(summary: pd.DataFrame) -> None:
    """Pretty print VAD summary by Guna."""
    print("\n" + "=" * 60)
    print("VAD SCORES BY GUNA")
    print("=" * 60)
    print("\n              Valence      Arousal      Dominance    N")
    print("-" * 60)

    for guna in ['Sattva', 'Rajas', 'Tamas']:
        if guna in summary.index:
            row = summary.loc[guna]
            print(f"{guna:10}   {row['V_mean']:+.3f}±{row['V_std']:.3f}   "
                  f"{row['A_mean']:+.3f}±{row['A_std']:.3f}   "
                  f"{row['D_mean']:+.3f}±{row['D_std']:.3f}   {int(row['count'])}")


# =============================================================================
# STEP 4: Main Execution
# =============================================================================

def main():
    """Main execution function."""

    # Configuration
    LEXICON_PATH = '/Users/sunny/Downloads/IOS/gunas/03_gap_analysis/VAD/unigrams-NRC-VAD-Lexicon-v2.1.txt'
    DATA_PATH = '/Users/sunny/Downloads/IOS/gunas/01_annotator_data/Guna_Annotations_NonFactual.csv'
    OUTPUT_PATH = '/Users/sunny/Downloads/IOS/gunas/03_gap_analysis/VAD/VAD_Results.csv'

    print("\n" + "#" * 60)
    print("  VAD (VALENCE-AROUSAL-DOMINANCE) ANALYSIS")
    print("#" * 60)

    # -----------------------------------------------------------------
    # Load Lexicon
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading NRC-VAD Lexicon")
    print("=" * 60)

    lexicon = load_vad_lexicon(LEXICON_PATH)
    print(f"Loaded {len(lexicon)} words from lexicon")

    # -----------------------------------------------------------------
    # Load Guna Data
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Loading Guna Annotations")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # Get unique sentences with majority Guna label
    sentences = df.groupby('Sentence_ID').agg({
        'Sentence_Text': 'first',
        'Label': lambda x: x.mode()[0]
    }).reset_index()
    sentences.columns = ['Sentence_ID', 'Text', 'Guna_Majority']

    print(f"Loaded {len(sentences)} unique sentences")

    # -----------------------------------------------------------------
    # Compute VAD Scores
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Computing VAD Scores")
    print("=" * 60)

    vad_results = []
    for _, row in sentences.iterrows():
        vad = compute_sentence_vad(row['Text'], lexicon)
        vad_results.append({
            'Sentence_ID': row['Sentence_ID'],
            'Text': row['Text'],
            'Guna_Majority': row['Guna_Majority'],
            'valence': vad['valence'],
            'arousal': vad['arousal'],
            'dominance': vad['dominance'],
            'matched_words': vad['matched_words'],
            'total_words': vad['total_words']
        })

    results_df = pd.DataFrame(vad_results)

    # Coverage stats
    matched = results_df['matched_words'].sum()
    total = results_df['total_words'].sum()
    print(f"Word coverage: {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"Sentences with VAD scores: {results_df['valence'].notna().sum()}/{len(results_df)}")

    # -----------------------------------------------------------------
    # Analyze by Guna
    # -----------------------------------------------------------------
    summary = analyze_vad_by_guna(results_df)
    print_vad_summary(summary)

    # -----------------------------------------------------------------
    # Correlations
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CORRELATIONS (Guna vs VAD)")
    print("=" * 60)
    print("\nGuna encoded as: Sattva=2, Rajas=1, Tamas=0")

    correlations = compute_correlations(results_df)
    for dim, vals in correlations.items():
        sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else ""
        print(f"  Guna vs {dim.capitalize():10}: r = {vals['r']:+.3f}, p = {vals['p']:.4f} {sig}")

    # -----------------------------------------------------------------
    # Key Findings
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Check if Gunas separate in VAD space
    sattva = results_df[results_df['Guna_Majority'] == 'Sattva']
    rajas = results_df[results_df['Guna_Majority'] == 'Rajas']
    tamas = results_df[results_df['Guna_Majority'] == 'Tamas']

    print("\nMean scores:")
    print(f"  Sattva: V={sattva['valence'].mean():+.3f}, A={sattva['arousal'].mean():+.3f}, D={sattva['dominance'].mean():+.3f}")
    print(f"  Rajas:  V={rajas['valence'].mean():+.3f}, A={rajas['arousal'].mean():+.3f}, D={rajas['dominance'].mean():+.3f}")
    print(f"  Tamas:  V={tamas['valence'].mean():+.3f}, A={tamas['arousal'].mean():+.3f}, D={tamas['dominance'].mean():+.3f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("-" * 60)

    v_corr = abs(correlations['valence']['r'])
    a_corr = abs(correlations['arousal']['r'])
    d_corr = abs(correlations['dominance']['r'])

    if max(v_corr, a_corr, d_corr) > 0.5:
        print("\n  Strong correlation with VAD dimension(s)")
        print("  → Gunas OVERLAP with emotional dimensions")
    elif max(v_corr, a_corr, d_corr) > 0.3:
        print("\n  Moderate correlation with VAD")
        print("  → Gunas PARTIALLY related to emotions")
    else:
        print("\n  Weak correlation with all VAD dimensions")
        print("  → Gunas capture something BEYOND emotions!")

    # -----------------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------------
    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 60 + "\n")

    return results_df, summary, correlations


if __name__ == "__main__":
    results_df, summary, correlations = main()
