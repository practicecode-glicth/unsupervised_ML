# 🛍️ K-Means Mall Customer Segmentation — Beginner Project

## 1. What is this project?

This project uses:

- K-Means Clustering
- Elbow Method
- Inertia
- StandardScaler
- Silhouette Score
- Joblib
- Streamlit

to segment mall customers into groups.

Example:

```text
Customer Data
      ↓
K-Means
      ↓
Cluster 0
Cluster 1
Cluster 2
...
```

---

## 2. Is training data required?

### YES

K-Means needs data to learn the cluster centers.

But K-Means is **unsupervised learning**, so it does NOT need a target/y column.

```text
Supervised:
X + y → Model → Prediction

K-Means:
X only → Model → Clusters
```

The Kaggle CSV is the **training data**.

---

## 3. Where is the training data?

Put the downloaded CSV here:

```text
data/Mall_Customers.csv
```

Download:
https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation

The project uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

---

## 4. Why two Python files?

### train.py

Responsible for training:

```text
CSV
 ↓
Features
 ↓
Scaling
 ↓
Elbow Method
 ↓
K-Means
 ↓
Save model
```

Run:

```bash
python train.py
```

### app.py

Responsible for using the trained model:

```text
Load .joblib
 ↓
Streamlit UI
 ↓
New customer
 ↓
Scale input
 ↓
Predict cluster
```

Run:

```bash
streamlit run app.py
```

---

## 5. Why Joblib?

After training, the model is saved:

```text
models/kmeans_model.joblib
```

Then app.py can load it:

```python
saved = joblib.load("models/kmeans_model.joblib")
```

This avoids retraining every time the app starts.

We save:

```text
model
scaler
features
K
```

The scaler is saved because new data must be transformed using the **same scaling rules** learned during training.

---

## 6. Libraries and why we install them

### Streamlit

```bash
pip install streamlit
```

Creates the web UI.

### Pandas

```bash
pip install pandas
```

Loads and handles CSV data.

### NumPy

```bash
pip install numpy
```

Handles numerical arrays.

### Scikit-learn

```bash
pip install scikit-learn
```

Provides:

```python
KMeans
StandardScaler
silhouette_score
```

### Matplotlib

```bash
pip install matplotlib
```

Creates the Elbow Method graph.

### Joblib

```bash
pip install joblib
```

Saves and loads the trained ML model.

Install everything:

```bash
pip install -r requirements.txt
```

---

## 7. Folder structure

```text
KMeans_Mall_Customer_Segmentation/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Mall_Customers.csv
│   └── README.txt
│
└── models/
    └── kmeans_model.joblib
```

`models/kmeans_model.joblib` appears after you run `train.py`.

---

## 8. Complete setup

### Step 1 — Download Kaggle data

Download the CSV and rename it:

```text
Mall_Customers.csv
```

Put it here:

```text
data/Mall_Customers.csv
```

### Step 2 — Install libraries

```bash
pip install -r requirements.txt
```

### Step 3 — Train

```bash
python train.py
```

This creates:

```text
models/kmeans_model.joblib
```

### Step 4 — Run Streamlit

```bash
streamlit run app.py
```

---

## 9. Training flow

```text
Kaggle Dataset
      ↓
Load CSV
      ↓
Select X features
      ↓
StandardScaler
      ↓
Elbow Method
      ↓
Choose K
      ↓
Train K-Means
      ↓
Silhouette Score
      ↓
Save model + scaler
```

---

## 10. Prediction flow

```text
New Customer
      ↓
Annual Income
Spending Score
      ↓
Same saved scaler
      ↓
K-Means
      ↓
Nearest learned cluster
      ↓
Cluster number
```

---

## 11. Important: fit_transform vs transform

During training:

```python
scaler.fit_transform(X)
```

The scaler learns from the training data.

For a new customer:

```python
scaler.transform(new_customer)
```

The existing scaler is reused.

Do NOT fit a new scaler on the new customer.

---

## 12. K-Means concepts used

### K

Number of clusters.

```text
K = 5
↓
5 clusters
```

### Centroid

Center/mean position of a cluster.

### Euclidean Distance

Used to determine which centroid is closest.

### Inertia

Measures total squared distance of points from their assigned centroids.

### Elbow Method

Helps choose a reasonable K.

### StandardScaler

Makes features comparable before distance calculations.

### Silhouette Score

Gives another indication of how well-separated clusters are.

---

## 13. Why K=5?

This project uses:

```python
BEST_K = 5
```

because it is a common teaching choice for the classic Mall Customers example.

In a real project, you should:

1. Look at the Elbow graph.
2. Compare several K values.
3. Check Silhouette Score.
4. Consider whether the resulting groups make business sense.

Do not blindly assume that K=5 is always correct.

---

## 14. Important understanding

K-Means does NOT know:

```text
Cluster 0 = Premium
Cluster 1 = Budget
```

It only knows:

```text
Cluster 0
Cluster 1
Cluster 2
...
```

We interpret the clusters using their average feature values.

---

## 15. Final mental model

```text
Training Data
      ↓
train.py
      ↓
Scale features
      ↓
Elbow Method
      ↓
Choose K
      ↓
K-Means
      ↓
Centroids + Clusters
      ↓
Save .joblib
      ↓
app.py
      ↓
Load model
      ↓
New Customer
      ↓
Same scaler
      ↓
Predict Cluster
```

## ⭐ Most important answer

**Yes, training data is required to train K-Means.**

**No, a target/y column is not required.**

**After training, the saved `.joblib` model can be reused by Streamlit without retraining every time.**
