"""
VADER Sentiment Analysis on Guna Annotations
============================================
Compares VADER sentiment scores with Guna classifications
to determine if Gunas capture something beyond sentiment.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n" + "=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)

df = pd.read_csv('/Users/sunny/Downloads/IOS/gunas/01_annotator_data/Guna_Annotations_NonFactual.csv')

# Get unique sentences with majority Guna label
sentences = df.groupby('Sentence_ID').agg({
    'Sentence_Text': 'first',
    'Label': lambda x: x.mode()[0]  # Majority vote
}).reset_index()
sentences.columns = ['Sentence_ID', 'Text', 'Guna_Majority']

print(f"Loaded {len(sentences)} unique sentences")
print(f"Guna distribution (majority vote):")
print(sentences['Guna_Majority'].value_counts())

# =============================================================================
# STEP 2: Run VADER on Each Sentence
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2: Running VADER")
print("=" * 50)

analyzer = SentimentIntensityAnalyzer()

vader_results = []
for _, row in sentences.iterrows():
    scores = analyzer.polarity_scores(row['Text'])
    vader_results.append({
        'Sentence_ID': row['Sentence_ID'],
        'Text': row['Text'],
        'Guna_Majority': row['Guna_Majority'],
        'vader_compound': scores['compound'],
        'vader_pos': scores['pos'],
        'vader_neg': scores['neg'],
        'vader_neu': scores['neu']
    })

results_df = pd.DataFrame(vader_results)
print(f"VADER analysis complete for {len(results_df)} sentences")

# =============================================================================
# STEP 3: Compare VADER Scores by Guna Category
# =============================================================================
print("\n" + "=" * 50)
print("STEP 3: VADER COMPOUND SCORE BY GUNA")
print("=" * 50)

summary = results_df.groupby('Guna_Majority')['vader_compound'].agg(['mean', 'std', 'min', 'max', 'count'])
print("\n")
print(summary.round(3))

# =============================================================================
# STEP 4: Detailed Analysis
# =============================================================================
print("\n" + "=" * 50)
print("STEP 4: DISTRIBUTION BY GUNA")
print("=" * 50)

for guna in ['Sattva', 'Rajas', 'Tamas']:
    subset = results_df[results_df['Guna_Majority'] == guna]['vader_compound']
    if len(subset) > 0:
        print(f"\n{guna} (n={len(subset)}):")
        print(f"  Range:  {subset.min():.3f} to {subset.max():.3f}")
        print(f"  Mean:   {subset.mean():.3f}")
        print(f"  Median: {subset.median():.3f}")
        print(f"  Std:    {subset.std():.3f}")

# =============================================================================
# STEP 5: Key Question Answer
# =============================================================================
print("\n" + "=" * 50)
print("STEP 5: KEY FINDINGS")
print("=" * 50)

sattva_mean = results_df[results_df['Guna_Majority'] == 'Sattva']['vader_compound'].mean()
rajas_mean = results_df[results_df['Guna_Majority'] == 'Rajas']['vader_compound'].mean()
tamas_mean = results_df[results_df['Guna_Majority'] == 'Tamas']['vader_compound'].mean()

print(f"\nMean VADER Compound Scores:")
print(f"  Sattva: {sattva_mean:.3f}")
print(f"  Rajas:  {rajas_mean:.3f}")
print(f"  Tamas:  {tamas_mean:.3f}")

# Interpretation
print("\n" + "-" * 50)
print("INTERPRETATION:")
print("-" * 50)

if sattva_mean > rajas_mean > tamas_mean:
    print("\n  Pattern: Sattva > Rajas > Tamas")
    print("  → Guna labels CORRELATE with sentiment")
    print("  → Gunas may partly reflect positive/negative tone")
elif abs(sattva_mean - tamas_mean) < 0.2:
    print("\n  Pattern: Similar scores across Gunas")
    print("  → Guna labels are INDEPENDENT of sentiment")
    print("  → Gunas capture something BEYOND sentiment!")
else:
    print("\n  Pattern: Mixed/Unexpected")
    print("  → Partial correlation with sentiment")
    print("  → Gunas capture more than just tone")

# Correlation analysis
guna_numeric = results_df['Guna_Majority'].map({'Sattva': 2, 'Rajas': 1, 'Tamas': 0})
correlation, p_value = stats.pearsonr(guna_numeric, results_df['vader_compound'])

print(f"\nCorrelation (Guna vs VADER): r = {correlation:.3f}, p = {p_value:.4f}")

# =============================================================================
# STEP 6: Sample Sentences
# =============================================================================
print("\n" + "=" * 50)
print("SAMPLE SENTENCES BY GUNA")
print("=" * 50)

for guna in ['Sattva', 'Rajas', 'Tamas']:
    subset = results_df[results_df['Guna_Majority'] == guna].head(3)
    print(f"\n{guna}:")
    for _, row in subset.iterrows():
        text_preview = row['Text'][:60] + "..." if len(row['Text']) > 60 else row['Text']
        print(f"  [{row['vader_compound']:+.3f}] {text_preview}")

# =============================================================================
# Save Results
# =============================================================================
results_df.to_csv('/Users/sunny/Downloads/IOS/gunas/03_gap_analysis/vader/VADER_Results.csv', index=False)
print("\n" + "=" * 50)
print(f"Results saved to: 03_gap_analysis/vader/VADER_Results.csv")
print("=" * 50 + "\n")
