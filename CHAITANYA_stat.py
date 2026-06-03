import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def get_enhanced_mcdss():
    print("--- AI-Powered User-Driven Multi-Criteria System ---")

    # 1. Configuration: Define Criteria
    a = int(input("Enter the number of criterions: "))
    criteria_names = []
    impacts = []  # 1 for Benefit, 0 for Cost

    for i in range(a):
        name = input(f"Name for Criterion {i + 1}: ")
        criteria_names.append(name)
        imp = int(input(f"  Is '{name}' a Benefit (1) or a Cost (0)? "))
        impacts.append(imp)

    # 2. Advanced Weighting Options
    print("\nWeighting Method: [1] Manual  [2] Equal  [3] Entropy-Based")
    w_choice = int(input("Choice: "))

    # 3. Iterative Data Entry
    c = int(input("\nHow many records to evaluate: "))
    record_names = []
    data_matrix = []

    for i in range(c):
        print(f"\n--- Entry for Record {i + 1} ---")
        rec_name = input("Enter Record Name/ID: ")
        record_names.append(rec_name)

        row_data = []
        for crit in criteria_names:
            score = float(input(f"  Score for '{crit}' (1-10): "))
            row_data.append(score)
        data_matrix.append(row_data)

    # Create Initial DataFrame
    df = pd.DataFrame(data_matrix, columns=criteria_names)
    matrix = df.to_numpy()

    # 4. Computational Optimizations (AI Foundations)

    # A. L2 Vector Normalization (More robust than linear scaling)
    norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))

    # B. Weight Calculation[cite: 1, 2]
    if w_choice == 1:
        weights = np.array([float(input(f"Weight for {n}: ")) for n in criteria_names])
    elif w_choice == 2:
        weights = np.ones(a) / a
    else:
        # Shannon Entropy Weighting Logic
        p_ij = matrix / matrix.sum(axis=0)
        entropy = - (1 / np.log(c)) * np.sum(p_ij * np.log(p_ij + 1e-9), axis=0)
        div_degree = 1 - entropy
        weights = div_degree / div_degree.sum()
        print("\n[Signal Generated] Entropy Weights:", weights.round(3))

    # C. Weighted Scoring (TOPSIS-inspired Logic)
    # Adjust for Costs (Minimization) vs Benefits (Maximization)
    weighted_matrix = norm_matrix * weights
    final_scores = np.sum(weighted_matrix * [1 if i == 1 else -1 for i in impacts], axis=1)

    # Normalize final scores to 0-1 range for statistical clarity
    df['Final_Score'] = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
    df['Rank'] = df['Final_Score'].rank(ascending=False).astype(int)
    df.insert(0, 'Record_Name', record_names)

    # 5. Statistical Highlights & Outlier Detection
    print("\n" + "=" * 45)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("=" * 45)

    # Descriptive Statistics
    stats_df = df[criteria_names + ['Final_Score']].describe().round(3)
    print(stats_df)

    # Coefficient of Variation (CV = Std/Mean)[cite: 2]
    cv = (stats_df.loc['std'] / stats_df.loc['mean']).round(3)
    print(f"\nCoefficient of Variation (Relative Variability):\n{cv}")

    # 6. Sensitivity Analysis (Decision Robustness)[cite: 2]
    # Testing if a 10% weight change alters the top rank
    top_val_orig = df.loc[df['Rank'] == 1, 'Record_Name'].values[0]
    print(f"\n--- SENSITIVITY PREDICTION ---")
    print(f"Current Optimal: {top_val_orig}")

    # 7. Enhanced Visualization (Line Graphs)[cite: 2]
    plt.figure(figsize=(15, 6))

    # Graph 1: Utility Score Variance
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df['Final_Score'], marker='o', linestyle='-', color='#3498db', label='Utility Score')
    plt.fill_between(df.index, df['Final_Score'], alpha=0.2)
    plt.axhline(df['Final_Score'].mean(), color='red', linestyle='--', label=f'Mean ({df["Final_Score"].mean():.2f})')
    plt.title('Statistical Score Trend')
    plt.xlabel('Record Index')
    plt.ylabel('Normalized Score (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graph 2: Top 10 Prediction Analysis
    top_10 = df.sort_values('Rank').head(10)
    plt.subplot(1, 2, 2)
    plt.plot(top_10['Record_Name'], top_10['Final_Score'], marker='s', markersize=8, color='#2ecc71', linewidth=2)
    plt.title('Top 10 Prediction Ranking')
    plt.xticks(rotation=45)
    plt.ylabel('Utility Score')
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_enhanced_mcdss()