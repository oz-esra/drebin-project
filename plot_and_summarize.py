"""
Plot and Summarize Advanced Experiment Results
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load Results
df_exp1 = pd.read_csv("experiment1_s1_s8_degradation.csv")
df_exp2 = pd.read_csv("experiment2_retraining_frequencies.csv")

print("="*60)
print("  EXPERIMENT 1: FEATURE GROUP DEGRADATION SUMMARY")
print("="*60)
f1_cols = [c for c in df_exp1.columns if c.endswith('_F1')]

summary_exp1 = []
for col in f1_cols:
    group_name = col.replace('_F1', '')
    start_f1 = df_exp1[col].iloc[0]
    end_f1 = df_exp1[col].iloc[-1]
    avg_f1 = df_exp1[col].mean()
    drop = start_f1 - end_f1
    summary_exp1.append({
        'Group': group_name,
        'Start F1 (2021-02)': round(start_f1, 3),
        'End F1 (2023-12)': round(end_f1, 3),
        'Drop': round(drop, 3),
        'Mean F1': round(avg_f1, 3)
    })

df_sum1 = pd.DataFrame(summary_exp1).sort_values('Drop', ascending=False)
print(df_sum1.to_string(index=False))

# Plot 1: Feature Group Degradation
plt.figure(figsize=(12, 6))
for col in f1_cols:
    plt.plot(df_exp1['Month'], df_exp1[col], marker='o', label=col.replace('_F1', ''))
plt.xticks(rotation=45)
plt.title("Drebin S1-S8 Feature Group Performance Degradation (Static Models)")
plt.xlabel("Month")
plt.ylabel("F1 Score")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("s1_s8_degradation_plot.png", dpi=300)
print("\nSaved plot: 's1_s8_degradation_plot.png'")

# Plot 2: Retraining Frequencies
plt.figure(figsize=(12, 6))
freq_cols = ['1M_Monthly_F1', '2M_Bimonthly_F1', '3M_Quarterly_F1', '6M_Semiannually_F1']
for col in freq_cols:
    plt.plot(df_exp2['Month'], df_exp2[col], marker='s', label=col.replace('_F1', ''))
plt.xticks(rotation=45)
plt.title("Retraining Frequency Comparison (1M vs 2M vs 3M vs 6M)")
plt.xlabel("Month")
plt.ylabel("F1 Score")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("retraining_frequency_plot.png", dpi=300)
print("Saved plot: 'retraining_frequency_plot.png'")  