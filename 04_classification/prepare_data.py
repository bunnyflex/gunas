"""
Prepare Guna Classification Data
================================
Creates guna_classifier_data.csv with majority vote labels
"""

import pandas as pd

# Load original annotations
df = pd.read_csv('/Users/sunny/Downloads/IOS/gunas/01_annotator_data/Guna_Annotations_NonFactual.csv')

# Get majority vote per sentence
sentences = df.groupby('Sentence_ID').agg({
    'Sentence_Text': 'first',
    'Label': lambda x: x.mode()[0]
}).reset_index()

# Encode labels: Tamas=0, Rajas=1, Sattva=2
label_map = {'Tamas': 0, 'Rajas': 1, 'Sattva': 2}
sentences['label'] = sentences['Label'].map(label_map)

# Create final dataset
classifier_data = sentences[['Sentence_Text', 'label']].copy()
classifier_data.columns = ['text', 'label']

# Save
output_path = '/Users/sunny/Downloads/IOS/gunas/04_classification/data/guna_classifier_data.csv'
classifier_data.to_csv(output_path, index=False)

# Report
print("=" * 50)
print("GUNA CLASSIFIER DATA PREPARED")
print("=" * 50)
print(f"\nTotal samples: {len(classifier_data)}")
print(f"\nClass distribution:")
print(f"  Tamas (0):  {(classifier_data['label'] == 0).sum()}")
print(f"  Rajas (1):  {(classifier_data['label'] == 1).sum()}")
print(f"  Sattva (2): {(classifier_data['label'] == 2).sum()}")
print(f"\nSaved to: {output_path}")
