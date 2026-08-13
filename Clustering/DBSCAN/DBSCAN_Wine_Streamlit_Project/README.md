# 🟢 DBSCAN Wine Dataset — Streamlit Mini Project

A beginner-friendly **Unsupervised Machine Learning** project using the Wine dataset and DBSCAN.

## 🎯 Project Goal

The application discovers groups of similar wine samples using **DBSCAN**.

The original Wine dataset has a class/target column, but **the target is intentionally NOT used for clustering**.

```text
Features → Used ✅
Target   → Not used ❌
```

This keeps the project genuinely unsupervised.

---

# 📊 Dataset

This project uses the **Wine dataset included with scikit-learn**.

You do NOT need to download a CSV from Kaggle.

The dataset is loaded with:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

It contains chemical measurements/features for wine samples.

Examples of features include:

```text
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
...
```

---

# 📁 Folder Structure

```text
DBSCAN_Wine_Streamlit_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
```

The `data/` folder is kept so you can later add your own CSV dataset.

---

# ⚙️ Install Libraries

Open the project folder in VS Code terminal:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

---

# ▶️ Run

```bash
streamlit run app.py
```

---

# 📚 Libraries

## Streamlit

```python
import streamlit as st
```

Creates the web application.

## Pandas

```python
import pandas as pd
```

Works with tabular data.

## NumPy

```python
import numpy as np
```

Performs numerical calculations.

## Scikit-learn

Used for:

```text
Wine dataset
StandardScaler
DBSCAN
PCA
Silhouette Score
```

## Matplotlib

Creates charts.

---

# 🔄 Complete Flow

```text
Wine Dataset
      ↓
Select Features
      ↓
StandardScaler
      ↓
DBSCAN
      ↓
Core / Border / Noise
      ↓
PCA Visualization
      ↓
Silhouette Score
      ↓
Cluster Profile
```

---

# 🟢 DBSCAN

DBSCAN means:

**Density-Based Spatial Clustering of Applications with Noise**

Easy meaning:

> Find dense areas and create clusters. Isolated points can become noise.

---

# ⭐ Important Parameters

## `eps`

Controls the neighborhood radius.

```python
DBSCAN(eps=1.0)
```

Easy meaning:

> How far should DBSCAN look around a point?

---

## `min_samples`

Controls the minimum number of nearby samples required for a dense region.

```python
DBSCAN(min_samples=5)
```

Easy meaning:

> How many nearby points do I need?

---

# 🧠 Core, Border and Noise

```text
Core
↓
Enough neighbors

Border
↓
Near a Core Point

Noise
↓
Not connected to a dense cluster
```

Noise is represented by:

```text
-1
```

---

# 📏 Why StandardScaler?

DBSCAN uses distance.

Different features can have very different ranges.

Therefore:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

This makes the features comparable before distance calculations.

---

# 📉 Why PCA?

The Wine dataset has many features.

A normal chart can only easily show 2 dimensions.

PCA converts the selected features into two components:

```text
PC1
PC2
```

for visualization.

### Important:

PCA is only used here for visualization.

DBSCAN still works on the scaled selected features.

---

# 📊 Silhouette Score

Silhouette Score helps evaluate cluster separation.

```text
Closer to +1 → Better separation
Around 0     → Overlap
Below 0      → Possible poor separation
```

It is not the only way to judge a DBSCAN result.

---

# 🆕 Testing New Data

The Streamlit app also lets you enter a new wine sample.

However, there is an important technical difference from supervised learning:

```python
model.predict(new_data)
```

is not available for normal scikit-learn DBSCAN.

Therefore, the project uses an educational approach:

```text
New Data
   ↓
Same StandardScaler
   ↓
Compare with existing Core Points
   ↓
Distance <= eps?
   ↓
Yes → approximately connected to a cluster
No  → possible Noise / Unknown
```

This is an **approximation for demonstration**, not a native DBSCAN prediction method.

---

# 🆚 DBSCAN vs Other Clustering Algorithms

| Algorithm | Main Idea |
|---|---|
| K-Means | Centroids |
| Hierarchical | Tree / Dendrogram |
| DBSCAN | Density |

Easy memory:

```text
K-Means       → Center
Hierarchical  → Tree
DBSCAN        → Density
```

---

# 💼 Possible Real-World Uses

DBSCAN can be useful for:

- Fraud/outlier detection
- GPS location clustering
- Customer behavior analysis
- Sensor anomaly detection
- Image processing
- Geographic data analysis
- Network anomaly detection

---

# ⚠️ Important Limitations

DBSCAN can struggle when:

1. `eps` is badly chosen.
2. Different clusters have very different densities.
3. Features are not scaled.
4. Data has very high dimensionality.

---

# 🚀 Improvements You Can Add

After understanding this version, you can add:

1. CSV upload
2. Automatic `eps` recommendation
3. K-distance graph
4. Compare DBSCAN with K-Means
5. Compare different `min_samples`
6. Download reports
7. Add interactive Plotly charts
8. Add a custom dataset option

---

# ⭐ Final Mental Model

```text
DBSCAN

eps
 ↓
How far do I look?

min_samples
 ↓
How many points do I need?

        ↓

Core Point
 ↓
Dense region
 ↓
Cluster

Border Point
 ↓
Near a Core

Noise
 ↓
Not connected to a dense region
```

## One-Line Definition

> **DBSCAN is an unsupervised clustering algorithm that creates clusters based on density and identifies isolated points as noise.**
