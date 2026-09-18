import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load exported monthly evaluation metrics
    df = pd.read_csv("monthly_degradation.csv")

    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot evaluation metrics across test timeline
    plt.plot(df['Month'], df['Precision'], marker='o', linewidth=2.5, label='Precision', color='#1f77b4')
    plt.plot(df['Month'], df['Recall'], marker='s', linewidth=2.5, label='Recall', color='#d62728')
    plt.plot(df['Month'], df['F1-Score'], marker='^', linewidth=2.5, label='F1-Score', color='#2ca02c')

    # Configure publication-grade plot parameters
    plt.title('TESSERACT Temporal Performance Degradation (Drebin 2021-2023)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Test Month (2023)', fontsize=11, labelpad=10)
    plt.ylabel('Score', fontsize=11, labelpad=10)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='lower left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure for thesis / paper inclusion
    output_filename = "monthly_degradation_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Publication-grade plot saved successfully as '{output_filename}'.")

if __name__ == "__main__":
    main()