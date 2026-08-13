# 🟢 DBSCAN Two Moons — Streamlit Mini Project

A very simple beginner-level DBSCAN project.

## Dataset

This project uses the built-in `make_moons` dataset from scikit-learn.

You do **not** need to download a CSV file.

```python
from sklearn.datasets import make_moons
```

The dataset contains only two features:

- Feature 1
- Feature 2

This makes DBSCAN easy to visualize.

## Why Two Moons?

The two groups are curved/irregular.

This helps demonstrate an important DBSCAN advantage:

> DBSCAN can find clusters based on density instead of requiring a center/centroid.

## Folder Structure

```text
DBSCAN_Two_Moons_Streamlit_Project/
│
├── app.py
├── requirements.txt
└── README.md
```

## Install

Open the folder in VS Code and run:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

## Run

```bash
streamlit run app.py
```

## What the project teaches

1. Unsupervised Learning
2. DBSCAN
3. `eps`
4. `min_samples`
5. Core Points
6. Border Points
7. Noise
8. StandardScaler
9. Silhouette Score
10. Cluster visualization
11. Testing a new point

## DBSCAN Mental Model

```text
eps
↓
How far should I look?

min_samples
↓
How many points do I need?

Enough nearby points
↓
Core Point
↓
Cluster

Near Core
↓
Border Point

Not connected
↓
Noise
```

## Important Note About New Data

Unlike supervised models, scikit-learn's DBSCAN does not provide a normal `.predict()` method for new unseen samples.

The project therefore includes an educational approximation that compares a new point with existing Core Points.

## Easy Comparison

```text
K-Means
→ Centroid

Hierarchical
→ Tree / Dendrogram

DBSCAN
→ Density
```
