"""
Plotting Comparison: Static Baseline vs. Cumulative Retraining (TESSERACT)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_static = pd.read_csv("monthly_degradation.csv")
df_retrain = pd.read_csv("monthly_retraining_results.csv")

# Ensure matching types
df_static['Month'] = df_static['Month'].astype(str)
df_retrain['Month'] = df_retrain['Month'].astype(str)

plt.figure(figsize=(10, 6))

# Plot Static Model Recall
plt.plot(df_static['Month'], df_static['Recall'], marker='o', color='#e74c3c', linewidth=2.5, label='Static Model (No Retraining)')

# Plot Retrained Model Recall
plt.plot(df_retrain['Month'], df_retrain['Recall'], marker='s', color='#2ecc71', linewidth=2.5, linestyle='--', label='Cumulative Retraining')

plt.title('TESSERACT Concept Drift Mitigation: Static vs. Retrained Model', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Evaluation Month (2023)', fontsize=12, labelpad=10)
plt.ylabel('Malware Recall (Detection Rate)', fontsize=12, labelpad=10)
plt.ylim(0, 1.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='lower left')
plt.tight_layout()

# Save publication-grade figure
plt.savefig('retraining_mitigation_plot.png', dpi=300)
print("Comparison plot successfully saved as 'retraining_mitigation_plot.png'.")