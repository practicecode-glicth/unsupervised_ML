# 🟣 GMM Customer Segmentation — Streamlit Mini Project

A beginner-friendly **Gaussian Mixture Model (GMM)** project for learning Unsupervised Machine Learning.

## 🎯 Project Goal

The project groups customers using:

- Annual Income
- Spending Score

GMM is different from K-Means because it gives a **probability** for each cluster.

Example:

```text
Cluster 0 → 10%
Cluster 1 → 25%
Cluster 2 → 65%
```

The customer is most likely in Cluster 2.

---

# 📁 Folder Structure

```text
GMM_Customer_Segmentation_Streamlit_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── customers.csv
```

---

# 📊 Dataset

The project includes a small, easy-to-understand customer dataset:

```text
Customer_ID
Annual_Income
Spending_Score
```

There is **no target column** because this is an unsupervised learning project.

The dataset is included in the project, so you do **not** need Kaggle or internet access to run it.

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

# ▶️ Run the Project

```bash
streamlit run app.py
```

---

# 🧠 What the Project Teaches

1. GMM
2. Unsupervised Learning
3. Soft Clustering
4. Gaussian Distribution
5. `n_components`
6. `predict()`
7. `predict_proba()`
8. StandardScaler
9. EM concept
10. AIC
11. BIC
12. Cluster visualization
13. New customer probability prediction
14. Downloadable CSV results

---

# 🔑 Important GMM Terms

## `n_components`

Number of Gaussian components.

```python
GaussianMixture(n_components=3)
```

means:

```text
Create 3 Gaussian components.
```

## `predict()`

Returns the most likely cluster.

## `predict_proba()`

Returns probabilities for every cluster.

Example:

```text
Cluster 0 → 15%
Cluster 1 → 20%
Cluster 2 → 65%
```

## AIC / BIC

Used as clues for choosing the number of components.

Generally:

```text
Lower AIC → Better
Lower BIC → Better
```

---

# 🆚 Easy Algorithm Memory

```text
K-Means      → Center
Hierarchical → Tree
DBSCAN       → Density
PCA          → Reduce
GMM          → Probability
```

---

# 💼 Possible Real-World Uses

GMM can be used for:

- Customer segmentation
- Market segmentation
- Image segmentation
- Pattern recognition
- Anomaly detection
- Speech recognition

---

# ⭐ One-Line Definition

> GMM is an unsupervised soft-clustering algorithm that uses multiple Gaussian distributions to calculate the probability of each data point belonging to each cluster.
