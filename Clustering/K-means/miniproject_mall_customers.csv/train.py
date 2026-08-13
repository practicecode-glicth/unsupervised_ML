"""
TRAIN.PY — Train the K-Means model.

Run:
    python train.py

Flow:
Kaggle CSV → features → scaling → Elbow Method → K-Means → save .joblib

K-Means is UNSUPERVISED:
X/features are required.
A target y column is NOT required.
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 1. File locations
DATA_PATH = "data/Mall_Customers.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.joblib")


# 2. Check training data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Training data not found. Put Mall_Customers.csv at: {DATA_PATH}"
    )


# 3. Load training data
# Pandas is used because CSV data is naturally handled as a DataFrame.
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())


# 4. Select X/features
# We do NOT select y because K-Means is unsupervised.
FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[FEATURES].copy()


# 5. Remove rows with missing selected features, if any
if X.isnull().sum().sum() > 0:
    X = X.dropna()
    print("Rows with missing selected features were removed.")


# 6. Scale features
# K-Means uses distance, so scaling is usually important.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. Elbow Method
# Test several K values and record inertia.
inertias = []
K_VALUES = range(2, 11)

for k in K_VALUES:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)


# 8. Draw and save the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(list(K_VALUES), inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True, alpha=0.2)
plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
plt.show()


# 9. Choose K
# K=5 is a common teaching choice for this dataset.
# In a real project, inspect the elbow and compare cluster quality.
BEST_K = 5


# 10. Train the final K-Means model
model = KMeans(
    n_clusters=BEST_K,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)


# 11. Silhouette Score
# This is an additional clue about cluster separation.
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")


# 12. Save model + scaler + feature order
# The scaler must also be saved so new data is transformed
# exactly like the training data.
package = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES,
    "k": BEST_K
}

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(package, MODEL_PATH)

print("Training completed.")
print("Saved model:", MODEL_PATH)
print("Saved graph: elbow_method.png")
