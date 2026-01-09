"""
Guna Annotations Inter-Rater Reliability Analysis
=================================================
Analyzes agreement among 3 annotators on Guna classifications (Sattva, Rajas, Tamas)

Sections:
1. Data Loading & Preprocessing
2. Fleiss' Kappa (Overall Agreement)
3. Pairwise Cohen's Kappa (Annotator Pairs)
4. Confusion Matrices (Disagreement Patterns)
5. Per-Category Agreement (Guna-specific reliability)
"""

import pandas as pd
import numpy as np
from itertools import combinations

# For statistical measures
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# SECTION 1: Data Loading & Preprocessing
# =============================================================================

def load_and_prepare_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV and pivot to get annotator columns side by side.

    Returns:
        raw_df: Original dataframe
        pivot_df: Pivoted with columns [Sentence_ID, Apoorva, Nidhi, Sunny]
    """
    raw_df = pd.read_csv(filepath)

    # Pivot so each row is one sentence with all 3 annotator labels
    pivot_df = raw_df.pivot(
        index='Sentence_ID',
        columns='Annotator',
        values='Label'
    ).reset_index()

    return raw_df, pivot_df


# =============================================================================
# SECTION 2: Fleiss' Kappa (Overall Agreement)
# =============================================================================

def compute_fleiss_kappa(pivot_df: pd.DataFrame, categories: list[str]) -> dict:
    """
    Compute Fleiss' Kappa for multi-rater agreement.

    Formula:
        κ = (P̄ - P̄e) / (1 - P̄e)

    Where:
        P̄  = mean proportion of agreeing pairs per subject
        P̄e = expected agreement by chance
    """
    annotators = ['Apoorva', 'Nidhi', 'Sunny']
    n_subjects = len(pivot_df)       # Number of sentences
    n_raters = len(annotators)       # 3 raters
    n_categories = len(categories)   # 3 categories (Sattva, Rajas, Tamas)

    # Build count matrix: rows = subjects, cols = categories
    # Each cell = number of raters who assigned that category to that subject
    count_matrix = np.zeros((n_subjects, n_categories))

    for i, row in pivot_df.iterrows():
        for j, cat in enumerate(categories):
            count = sum(1 for ann in annotators if row[ann] == cat)
            count_matrix[i, j] = count

    # P_i = proportion of agreeing pairs for subject i
    # P_i = (1 / n(n-1)) * sum(n_ij^2) - n
    P_i = np.zeros(n_subjects)
    for i in range(n_subjects):
        sum_sq = np.sum(count_matrix[i, :] ** 2)
        P_i[i] = (sum_sq - n_raters) / (n_raters * (n_raters - 1))

    # P̄ = mean of all P_i
    P_bar = np.mean(P_i)

    # p_j = proportion of all ratings in category j
    p_j = np.sum(count_matrix, axis=0) / (n_subjects * n_raters)

    # P̄e = sum of p_j^2 (expected agreement by chance)
    P_e_bar = np.sum(p_j ** 2)

    # Fleiss' Kappa
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar) if (1 - P_e_bar) != 0 else 0

    return {
        'kappa': kappa,
        'observed_agreement': P_bar,
        'expected_agreement': P_e_bar,
        'category_proportions': dict(zip(categories, p_j)),
        'n_subjects': n_subjects,
        'n_raters': n_raters
    }


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) interpretation guidelines."""
    if kappa < 0:
        return "Poor"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


# =============================================================================
# SECTION 3: Pairwise Cohen's Kappa (Annotator Pairs)
# =============================================================================

def compute_pairwise_kappa(pivot_df: pd.DataFrame) -> dict:
    """
    Compute Cohen's Kappa for each annotator pair.

    Cohen's Kappa is for 2 raters only, so we compute it for:
        - Apoorva vs Nidhi
        - Apoorva vs Sunny
        - Nidhi vs Sunny
    """
    annotators = ['Apoorva', 'Nidhi', 'Sunny']
    pairs = list(combinations(annotators, 2))

    results = {}

    for ann1, ann2 in pairs:
        labels1 = pivot_df[ann1].values
        labels2 = pivot_df[ann2].values

        kappa = cohen_kappa_score(labels1, labels2)

        # Simple percent agreement
        agreement_count = np.sum(labels1 == labels2)
        percent_agreement = agreement_count / len(labels1) * 100

        pair_name = f"{ann1} vs {ann2}"
        results[pair_name] = {
            'kappa': kappa,
            'interpretation': interpret_kappa(kappa),
            'percent_agreement': percent_agreement,
            'agreements': agreement_count,
            'disagreements': len(labels1) - agreement_count
        }

    return results


def print_pairwise_results(pairwise_results: dict) -> None:
    """Pretty print pairwise kappa results."""
    print("\n" + "=" * 60)
    print("PAIRWISE COHEN'S KAPPA")
    print("=" * 60)

    for pair, data in pairwise_results.items():
        print(f"\n{pair}:")
        print(f"  Kappa:            {data['kappa']:.4f} ({data['interpretation']})")
        print(f"  % Agreement:      {data['percent_agreement']:.1f}%")
        print(f"  Agreed/Disagreed: {data['agreements']}/{data['disagreements']}")


# =============================================================================
# SECTION 4: Confusion Matrices (Disagreement Patterns)
# =============================================================================

def compute_confusion_matrices(pivot_df: pd.DataFrame, categories: list[str]) -> dict:
    """
    Compute confusion matrix for each annotator pair.

    Shows which labels get confused with which.
    Rows = Annotator 1's labels, Cols = Annotator 2's labels
    """
    annotators = ['Apoorva', 'Nidhi', 'Sunny']
    pairs = list(combinations(annotators, 2))

    results = {}

    for ann1, ann2 in pairs:
        labels1 = pivot_df[ann1].values
        labels2 = pivot_df[ann2].values

        cm = confusion_matrix(labels1, labels2, labels=categories)

        pair_name = f"{ann1} vs {ann2}"
        results[pair_name] = {
            'matrix': cm,
            'annotator1': ann1,
            'annotator2': ann2
        }

    return results


def plot_confusion_matrices(confusion_results: dict, categories: list[str],
                            save_path: str = None) -> None:
    """
    Plot confusion matrices as heatmaps for each annotator pair.
    """
    n_pairs = len(confusion_results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))

    if n_pairs == 1:
        axes = [axes]

    for ax, (pair_name, data) in zip(axes, confusion_results.items()):
        cm = data['matrix']
        ann1 = data['annotator1']
        ann2 = data['annotator2']

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories,
            ax=ax,
            cbar=False
        )

        ax.set_xlabel(f"{ann2}'s Labels")
        ax.set_ylabel(f"{ann1}'s Labels")
        ax.set_title(f"{pair_name}")

    plt.suptitle("Confusion Matrices: Annotator Disagreement Patterns",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrices to: {save_path}")

    plt.show()


def analyze_disagreements(confusion_results: dict, categories: list[str]) -> dict:
    """
    Identify most common disagreement patterns across all pairs.
    """
    disagreement_counts = {}

    for pair_name, data in confusion_results.items():
        cm = data['matrix']

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i != j:  # Only off-diagonal (disagreements)
                    pattern = f"{cat1} ↔ {cat2}"
                    reverse_pattern = f"{cat2} ↔ {cat1}"

                    # Combine bidirectional disagreements
                    if reverse_pattern in disagreement_counts:
                        disagreement_counts[reverse_pattern] += cm[i, j]
                    else:
                        if pattern not in disagreement_counts:
                            disagreement_counts[pattern] = 0
                        disagreement_counts[pattern] += cm[i, j]

    # Sort by frequency
    sorted_disagreements = dict(
        sorted(disagreement_counts.items(), key=lambda x: x[1], reverse=True)
    )

    return sorted_disagreements


# =============================================================================
# SECTION 5: Per-Category Agreement (Guna-specific reliability)
# =============================================================================

def compute_per_category_agreement(pivot_df: pd.DataFrame,
                                   categories: list[str]) -> dict:
    """
    Compute agreement metrics for each Guna category separately.

    For each category, calculates:
        - How often all 3 annotators agreed on that label
        - How often 2 out of 3 agreed
        - Category-specific Fleiss' Kappa (binary: this category vs others)
    """
    annotators = ['Apoorva', 'Nidhi', 'Sunny']
    n_subjects = len(pivot_df)

    results = {}

    for cat in categories:
        # Count agreement levels for this category
        full_agreement = 0      # All 3 chose this label
        partial_agreement = 0   # Exactly 2 chose this label
        no_agreement = 0        # 0 or 1 chose this label

        # Track when category was assigned
        times_assigned = 0

        for _, row in pivot_df.iterrows():
            votes = [row[ann] for ann in annotators]
            count = votes.count(cat)
            times_assigned += count

            if count == 3:
                full_agreement += 1
            elif count == 2:
                partial_agreement += 1
            else:
                no_agreement += 1

        # Binary Fleiss' Kappa for this category
        # Convert to binary: 1 if annotator chose this category, 0 otherwise
        binary_kappa = compute_binary_fleiss_kappa(pivot_df, annotators, cat)

        results[cat] = {
            'full_agreement_3of3': full_agreement,
            'partial_agreement_2of3': partial_agreement,
            'low_agreement_0or1': no_agreement,
            'total_assignments': times_assigned,
            'avg_per_sentence': times_assigned / n_subjects,
            'binary_kappa': binary_kappa,
            'binary_kappa_interpretation': interpret_kappa(binary_kappa)
        }

    return results


def compute_binary_fleiss_kappa(pivot_df: pd.DataFrame,
                                 annotators: list[str],
                                 target_category: str) -> float:
    """
    Compute Fleiss' Kappa treating one category as positive, rest as negative.

    This shows how reliably annotators identify a specific Guna.
    """
    n_subjects = len(pivot_df)
    n_raters = len(annotators)

    # Binary: count of raters who chose target category per subject
    counts = []
    for _, row in pivot_df.iterrows():
        count = sum(1 for ann in annotators if row[ann] == target_category)
        counts.append(count)

    counts = np.array(counts)

    # For binary case: categories are [target, not-target]
    # n_ij for target = counts, for not-target = n_raters - counts

    # P_i for each subject
    P_i = (counts**2 + (n_raters - counts)**2 - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # p_j proportions
    p_target = np.sum(counts) / (n_subjects * n_raters)
    p_other = 1 - p_target

    P_e_bar = p_target**2 + p_other**2

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar) if (1 - P_e_bar) != 0 else 0

    return kappa


def print_per_category_results(category_results: dict) -> None:
    """Pretty print per-category agreement results."""
    print("\n" + "=" * 60)
    print("PER-CATEGORY AGREEMENT (Guna-specific)")
    print("=" * 60)

    for cat, data in category_results.items():
        print(f"\n{cat.upper()}:")
        print(f"  Full Agreement (3/3):    {data['full_agreement_3of3']} sentences")
        print(f"  Partial Agreement (2/3): {data['partial_agreement_2of3']} sentences")
        print(f"  Low Agreement (0-1/3):   {data['low_agreement_0or1']} sentences")
        print(f"  Total Assignments:       {data['total_assignments']}")
        print(f"  Binary Kappa:            {data['binary_kappa']:.4f} ({data['binary_kappa_interpretation']})")


# =============================================================================
# SECTION 6: Main Execution
# =============================================================================

def print_fleiss_results(fleiss_results: dict) -> None:
    """Pretty print Fleiss' Kappa results."""
    print("\n" + "=" * 60)
    print("FLEISS' KAPPA (Overall Multi-Rater Agreement)")
    print("=" * 60)

    print(f"\n  Kappa:               {fleiss_results['kappa']:.4f}")
    print(f"  Interpretation:      {interpret_kappa(fleiss_results['kappa'])}")
    print(f"  Observed Agreement:  {fleiss_results['observed_agreement']:.4f}")
    print(f"  Expected (Chance):   {fleiss_results['expected_agreement']:.4f}")
    print(f"\n  Subjects (sentences): {fleiss_results['n_subjects']}")
    print(f"  Raters:               {fleiss_results['n_raters']}")

    print("\n  Category Proportions:")
    for cat, prop in fleiss_results['category_proportions'].items():
        print(f"    {cat}: {prop:.3f} ({prop*100:.1f}%)")


def print_disagreement_summary(disagreements: dict) -> None:
    """Pretty print disagreement patterns."""
    print("\n" + "=" * 60)
    print("DISAGREEMENT PATTERNS (Most Common)")
    print("=" * 60)

    for pattern, count in disagreements.items():
        print(f"  {pattern}: {count} instances")


def main():
    """Main execution function."""

    # Configuration
    FILEPATH = '/Users/sunny/Downloads/IOS/gunas/01_annotator_data/Guna_Annotations_NonFactual.csv'
    CATEGORIES = ['Sattva', 'Rajas', 'Tamas']
    SAVE_PLOTS = True
    PLOT_PATH = '/Users/sunny/Downloads/IOS/gunas/02_annotator_agreement/confusion_matrices.png'

    print("\n" + "#" * 60)
    print("  GUNA ANNOTATIONS - INTER-RATER RELIABILITY ANALYSIS")
    print("#" * 60)

    # -----------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------
    print("\nLoading data...")
    raw_df, pivot_df = load_and_prepare_data(FILEPATH)
    print(f"Loaded {len(pivot_df)} sentences with 3 annotators each.")

    # -----------------------------------------------------------------
    # 2. Fleiss' Kappa (Overall Agreement)
    # -----------------------------------------------------------------
    fleiss_results = compute_fleiss_kappa(pivot_df, CATEGORIES)
    print_fleiss_results(fleiss_results)

    # -----------------------------------------------------------------
    # 3. Pairwise Cohen's Kappa
    # -----------------------------------------------------------------
    pairwise_results = compute_pairwise_kappa(pivot_df)
    print_pairwise_results(pairwise_results)

    # -----------------------------------------------------------------
    # 4. Confusion Matrices
    # -----------------------------------------------------------------
    confusion_results = compute_confusion_matrices(pivot_df, CATEGORIES)
    disagreements = analyze_disagreements(confusion_results, CATEGORIES)
    print_disagreement_summary(disagreements)

    if SAVE_PLOTS:
        plot_confusion_matrices(confusion_results, CATEGORIES, PLOT_PATH)
    else:
        plot_confusion_matrices(confusion_results, CATEGORIES)

    # -----------------------------------------------------------------
    # 5. Per-Category Agreement
    # -----------------------------------------------------------------
    category_results = compute_per_category_agreement(pivot_df, CATEGORIES)
    print_per_category_results(category_results)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Overall Fleiss' Kappa: {fleiss_results['kappa']:.4f} ({interpret_kappa(fleiss_results['kappa'])})")
    print("\n  Pairwise Agreement:")
    for pair, data in pairwise_results.items():
        print(f"    {pair}: κ={data['kappa']:.4f}")
    print("\n  Most Reliable Category:")
    best_cat = max(category_results.items(), key=lambda x: x[1]['binary_kappa'])
    print(f"    {best_cat[0]} (κ={best_cat[1]['binary_kappa']:.4f})")
    print("\n  Most Confused Labels:")
    top_confusion = list(disagreements.items())[0]
    print(f"    {top_confusion[0]}: {top_confusion[1]} instances")

    print("\n" + "#" * 60)
    print("  ANALYSIS COMPLETE")
    print("#" * 60 + "\n")

    return {
        'fleiss': fleiss_results,
        'pairwise': pairwise_results,
        'confusion': confusion_results,
        'disagreements': disagreements,
        'per_category': category_results
    }


if __name__ == "__main__":
    results = main()
