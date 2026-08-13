# 🌳 Hierarchical Clustering — Mall Customer Segmentation

A beginner-friendly mini project using **Kaggle + Python + Scikit-learn + SciPy + Streamlit**.

---

## 🎯 Project Goal

We will use customer information to discover natural customer groups.

The project uses:

- Age
- Annual Income
- Spending Score

The algorithm is:

**Agglomerative Hierarchical Clustering**

---

# 📊 Dataset

Use the classic **Mall Customer Segmentation Data** from Kaggle.

Kaggle examples using this dataset include customer segmentation and hierarchical clustering work.

Kaggle search:
https://www.kaggle.com/datasets

Search for:

```text
Mall Customer Segmentation Data
```

Download the CSV and rename it:

```text
Mall_Customers.csv
```

Put it inside:

```text
data/Mall_Customers.csv
```

---

# 📁 Folder Structure

```text
Hierarchical_Mall_Customer_Streamlit_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── Mall_Customers.csv
```

---

# 🧠 Is training data required?

## YES ✅

The algorithm needs data to discover the clusters.

But this is **Unsupervised Learning**.

Therefore:

```text
X / Features → Required ✅

y / Target   → Not required ❌
```

For this project:

```text
Age
Annual Income
Spending Score
```

are the features.

There is no target such as:

```text
Customer Type
```

---

# ⚙️ Installation

Open the project folder in VS Code terminal.

Run:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install streamlit pandas numpy scikit-learn scipy matplotlib
```

---

# ▶️ Run the Project

Run:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

---

# 📚 What Each Library Does

## Streamlit

```python
import streamlit as st
```

Creates the web application.

---

## Pandas

```python
import pandas as pd
```

Loads and manages the CSV dataset.

---

## NumPy

```python
import numpy as np
```

Provides numerical operations.

---

## Scikit-learn

Used for:

```python
StandardScaler
AgglomerativeClustering
silhouette_score
```

---

## SciPy

Used for:

```python
linkage
dendrogram
```

SciPy creates the hierarchical structure and dendrogram.

---

## Matplotlib

Creates:

- Dendrogram
- Cluster distribution chart
- Customer segmentation chart

---

# 🔄 Complete Project Flow

```text
Kaggle Dataset
      ↓
Load CSV
      ↓
Select Features
      ↓
StandardScaler
      ↓
Dendrogram
      ↓
Choose Number of Clusters
      ↓
Agglomerative Clustering
      ↓
Cluster Labels
      ↓
Silhouette Score
      ↓
Cluster Visualization
      ↓
Business Interpretation
```

---

# 🌳 What is Agglomerative Clustering?

It is a bottom-up hierarchical clustering technique.

Initially:

```text
[A] [B] [C] [D] [E]
```

Then:

```text
[AB] [C] [DE]
```

Then:

```text
[ABC] [DE]
```

Finally:

```text
[ABCDE]
```

It keeps merging clusters based on similarity/distance.

---

# 🔗 What is Linkage?

Linkage decides how the distance between clusters is calculated.

This project uses:

```python
linkage="ward"
```

Ward linkage tries to create compact clusters by minimizing the increase in within-cluster variance.

Other common linkage methods:

```text
single
complete
average
ward
```

---

# 🌳 What is a Dendrogram?

A dendrogram is a tree diagram showing the history of cluster merging.

It helps us understand:

- Which points/clusters merged
- At what distance they merged
- Where we can cut the hierarchy

---

# 📏 Why do we use StandardScaler?

Hierarchical Clustering is distance-based.

Suppose:

```text
Age:              18–70

Annual Income:    15–150
```

Income has a larger numerical range.

Without scaling, it can have too much influence on distance.

So we use:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 📊 Silhouette Score

The project calculates:

```python
silhouette_score(X_scaled, labels)
```

Easy interpretation:

```text
Closer to +1 → Better separated
Around 0     → Overlapping clusters
Below 0      → Possible poor assignments
```

It is an evaluation metric, not a magic "correctness" score.

---

# 💼 Business Use Case

After clustering customers, a business could create strategies for different groups.

Example:

```text
High Income + High Spending
→ Premium offers

Low Income + Low Spending
→ Budget offers

High Income + Low Spending
→ Engagement campaigns

Low Income + High Spending
→ Loyalty campaigns
```

The exact interpretation depends on the business.

---

# 🧠 Important Difference from K-Means

### K-Means

```text
Choose K
 ↓
Centroids
 ↓
Distance
 ↓
Assign
 ↓
New Centroids
 ↓
Repeat
```

### Hierarchical

```text
Data
 ↓
Distance
 ↓
Merge
 ↓
Merge
 ↓
Hierarchy
 ↓
Dendrogram
 ↓
Cut
 ↓
Clusters
```

---

# ⭐ Technical Words to Know

| Word | Easy Meaning |
|---|---|
| Unsupervised Learning | No target/y column |
| Clustering | Making groups |
| Agglomerative | Bottom-up merging |
| Hierarchical | Creates levels/tree |
| Distance | How far points are |
| Linkage | How cluster distance is calculated |
| Ward | Linkage method for compact clusters |
| Dendrogram | Tree showing merges |
| StandardScaler | Scales features |
| Silhouette Score | Measures cluster separation |
| Cluster Label | Group number assigned to a point |

---

# 🚀 Possible Improvements

After understanding this basic project, you can improve it by adding:

1. Upload your own CSV
2. Dynamic feature selection
3. Automatic cluster recommendation
4. Compare Ward / Complete / Average linkage
5. Compare K-Means vs Hierarchical
6. Add PCA visualization
7. Add customer search
8. Add downloadable cluster reports

---

# ⭐ Final Mental Model

Remember:

```text
Hierarchical Clustering

Individual Points
       ↓
Calculate Distance
       ↓
Find Closest
       ↓
Merge
       ↓
Merge Again
       ↓
Build Hierarchy
       ↓
Dendrogram 🌳
       ↓
Cut
       ↓
Final Clusters
```

This is an **unsupervised learning** project, so:

```text
Training data → YES ✅

Target/y → NO ❌
```
