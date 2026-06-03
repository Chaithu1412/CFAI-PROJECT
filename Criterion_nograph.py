import numpy as np

print("--- Welcome to Multi-Criteria System ---")

# 1. Define Criteria
a = int(input("Tell me the number of criterions: "))
z = []
for i in range(a):
    b = input(f"Enter the name for Criterion {i+1}: ")
    z.append(b)

# 2. Define Weights (How important is each criterion?)
print("\nDefine the importance (0.0 to 1.0) for each criterion.")
weights = []
for i in range(a):
    w = float(input(f"Weight for {z[i]}: "))
    weights.append(w)

# 3. Collect Records
c = int(input("\nTell me the number of records to be stored: "))
record_names = []
data_matrix = []

for i in range(c):
    name = input(f"\nEnter name/ID for Record {i+1}: ")
    record_names.append(name)
    data = []
    for j in range(a):
        # Taking input based on your criteria list
        d = float(input(f"  Enter score for {z[j]} (1-10): "))
        data.append(d)
    data_matrix.append(data)

# --- COMPUTATIONAL CORE ---

# Convert to Numpy Array for optimized math
matrix = np.array(data_matrix)
weights = np.array(weights)

# Step 1: Normalization (Scaling 1-10 down to 0-1)
normalized_matrix = matrix / 10.0

# Step 2: Weighted Calculation
# We multiply each row by the weights and sum them (Matrix-Vector Multiplication)
final_scores = np.dot(normalized_matrix, weights)

# --- OUTPUT RESULTS ---

print("\n" + "="*30)
print("FINAL EVALUATION MATRIX")
print("="*30)

# Print Header
print(f"{'Record':<12}", end="")
for crit in z:
    print(f"{crit:<10}", end="")
print(f"{'TOTAL SCORE'}")

# Print Data Rows+++++++++++++
for i in range(c):
    print(f"{record_names[i]:<12}", end="")
    for j in range(a):
        print(f"{data_matrix[i][j]:<10}", end="")
    print(f"{final_scores[i]:.4f}")

print("-" * 30)

# Step 3: Choose the Best Record
best_index = np.argmax(final_scores)
print(f"DECISION: The best record chosen is '{record_names[best_index]}' with a score of {final_scores[best_index]:.4f}")