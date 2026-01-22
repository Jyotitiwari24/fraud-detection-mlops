import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
print("Loading data...")
df = pd.read_csv('creditcard_data.csv')

# Basic info
print("\n" + "="*50)
print("BASIC INFORMATION")
print("="*50)
print(f"Total transactions: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print(f"\nColumns: {list(df.columns)}")

# Check for missing data
print("\n" + "="*50)
print("MISSING DATA CHECK")
print("="*50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing data! Good to go.")
else:
    print("Missing values:")
    print(missing[missing > 0])

# Look at fraud vs legitimate
print("\n" + "="*50)
print("FRAUD vs LEGITIMATE")
print("="*50)
fraud_count = df['Class'].value_counts()
print(fraud_count)
print(f"\nFraud percentage: {(fraud_count[1]/len(df))*100:.2f}%")

# Statistics for fraud vs legitimate
print("\n" + "="*50)
print("TRANSACTION AMOUNTS")
print("="*50)
print("\nLegitimate transactions:")
print(df[df['Class']==0]['Amount'].describe())
print("\nFraudulent transactions:")
print(df[df['Class']==1]['Amount'].describe())

# Create visualizations
print("\n" + "="*50)
print("Creating visualizations...")
print("="*50)

# Set style
sns.set_style("whitegrid")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Class distribution
axes[0, 0].bar(['Legitimate', 'Fraud'], fraud_count.values, color=['green', 'red'])
axes[0, 0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(fraud_count.values):
    axes[0, 0].text(i, v + 500, str(v), ha='center', fontweight='bold')

# Plot 2: Amount distribution
axes[0, 1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.5, label='Legitimate', color='green')
axes[0, 1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.5, label='Fraud', color='red')
axes[0, 1].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Amount ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].set_xlim([0, 500])  # Zoom in to see patterns

# Plot 3: Time distribution
axes[1, 0].hist(df[df['Class']==0]['Time'], bins=50, alpha=0.5, label='Legitimate', color='green')
axes[1, 0].hist(df[df['Class']==1]['Time'], bins=50, alpha=0.5, label='Fraud', color='red')
axes[1, 0].set_title('Transaction Time Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Plot 4: Feature V1 distribution 
axes[1, 1].hist(df[df['Class']==0]['V1'], bins=50, alpha=0.5, label='Legitimate', color='green')
axes[1, 1].hist(df[df['Class']==1]['V1'], bins=50, alpha=0.5, label='Fraud', color='red')
axes[1, 1].set_title('Feature V1 Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('V1 Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to 'data_exploration.png'")

# Key insights
print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)
print("1. We have IMBALANCED data - way more legitimate than fraud")
print("   → We'll need special techniques to handle this")
print("\n2. Fraud transactions have different patterns:")
print(f"   - Average fraud amount: ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"   - Average legit amount: ${df[df['Class']==0]['Amount'].mean():.2f}")
print("\n3. Features V1-V28 are anonymized (privacy)")
print("   → But they still have predictive power")

print("\n✓ Data exploration complete!")
print("Next step: Build the model")