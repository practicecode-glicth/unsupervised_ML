"""
Hierarchical Clustering — Mall Customer Segmentation
=====================================================

Mini project using:
- Kaggle Mall Customers dataset
- Agglomerative Hierarchical Clustering
- StandardScaler
- Ward Linkage
- Dendrogram
- Silhouette Score
- Streamlit

Run:
    streamlit run app.py

IMPORTANT:
This is an UNSUPERVISED learning project.
There is NO target/y column required.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Streamlit page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hierarchical Customer Segmentation",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Hierarchical Clustering — Mall Customer Segmentation")
st.write(
    "A beginner-friendly Unsupervised ML project using "
    "Agglomerative Hierarchical Clustering."
)


# ---------------------------------------------------------
# 2. Explain the project flow
# ---------------------------------------------------------
with st.expander("🧠 Understand the complete project flow"):
    st.markdown("""
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
Business Interpretation
```

### Important:
Hierarchical Clustering is **unsupervised learning**.

Therefore:

```text
X / Features → Required ✅
y / Target    → Not required ❌
```
""")


# ---------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------
DATA_PATH = "data/Mall_Customers.csv"

st.sidebar.header("📂 Dataset")

if not os.path.exists(DATA_PATH):
    st.error(
        "Mall_Customers.csv was not found. "
        "Download the Kaggle dataset and put it inside data/."
    )

    st.info(
        "Expected path: data/Mall_Customers.csv"
    )

    st.stop()

# Pandas reads the Kaggle CSV into a DataFrame.
df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# 4. Show dataset
# ---------------------------------------------------------
st.header("1️⃣ Dataset")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Rows", len(df))

with c2:
    st.metric("Columns", len(df.columns))

with c3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# 5. Select features
# ---------------------------------------------------------
st.header("2️⃣ Select Features")

# These columns exist in the classic Mall Customers dataset.
# We use numeric features because distance-based clustering
# works naturally with numerical values.
available_features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

existing_features = [
    col for col in available_features
    if col in df.columns
]

if len(existing_features) < 2:
    st.error(
        "The dataset does not contain enough expected numeric features."
    )
    st.stop()

selected_features = st.multiselect(
    "Choose at least 2 features:",
    existing_features,
    default=existing_features
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

# X contains only the selected input features.
X = df[selected_features].copy()

# Remove rows with missing values in selected columns.
X = X.dropna()

st.write("Selected features:", selected_features)


# ---------------------------------------------------------
# 6. StandardScaler
# ---------------------------------------------------------
st.header("3️⃣ Feature Scaling")

st.write("""
Hierarchical Clustering uses distance.

If one feature has a much larger numerical range than another,
it can dominate the distance calculation.

StandardScaler changes features approximately to:

```text
mean = 0
standard deviation = 1
```
""")

scaler = StandardScaler()

# fit_transform() learns scaling information from the dataset
# and then transforms the dataset.
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 7. Dendrogram
# ---------------------------------------------------------
st.header("4️⃣ Dendrogram 🌳")

st.write("""
The dendrogram shows the hierarchy of cluster merging.

To understand it:

- Lower merge height → clusters are more similar.
- Higher merge height → clusters are less similar.
- A horizontal cut can be used to decide the final number of clusters.
""")

# scipy's linkage() creates the hierarchical merge information.
# method="ward" means Ward linkage is used.
linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    ax=ax,
    no_labels=True,
    color_threshold=None
)

ax.set_title("Hierarchical Clustering Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance / Merge Height")

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# 8. Choose number of clusters
# ---------------------------------------------------------
st.header("5️⃣ Choose Number of Clusters")

n_clusters = st.slider(
    "Number of final clusters:",
    min_value=2,
    max_value=8,
    value=5
)

st.write(
    f"You selected **{n_clusters} clusters**."
)


# ---------------------------------------------------------
# 9. Agglomerative Clustering
# ---------------------------------------------------------
st.header("6️⃣ Train Hierarchical Clustering Model")

st.write("""
We use **Agglomerative Clustering**.

It works from bottom to top:

```text
Individual points
       ↓
Find closest clusters
       ↓
Merge
       ↓
Find closest clusters again
       ↓
Merge
       ↓
Continue
```
""")

# Create the Agglomerative Clustering model.
#
# n_clusters:
#     Number of final groups.
#
# linkage="ward":
#     Uses Ward's method to create compact clusters.
#
# metric:
#     Ward linkage uses Euclidean distance.
model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

# fit_predict() learns the cluster structure and
# returns a cluster label for every row.
labels = model.fit_predict(X_scaled)


# ---------------------------------------------------------
# 10. Add labels to the data
# ---------------------------------------------------------
result = X.copy()

result["Cluster"] = labels

st.success("✅ Clustering completed!")


# ---------------------------------------------------------
# 11. Silhouette Score
# ---------------------------------------------------------
st.header("7️⃣ Silhouette Score")

score = silhouette_score(
    X_scaled,
    labels
)

st.metric(
    "Silhouette Score",
    f"{score:.3f}"
)

st.write("""
### Easy meaning

Silhouette Score helps us understand how well-separated
the clusters are.

Generally:

```text
Closer to +1 → Better separated clusters
Around 0     → Clusters overlap
Below 0      → Some points may be in the wrong cluster
```

Use this as an evaluation clue, not as the only decision.
""")


# ---------------------------------------------------------
# 12. Cluster sizes
# ---------------------------------------------------------
st.header("8️⃣ Cluster Distribution")

cluster_counts = (
    result["Cluster"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Customers")
ax.set_title("Customers in Each Cluster")

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# 13. Two-feature visualization
# ---------------------------------------------------------
st.header("9️⃣ Cluster Visualization")

if (
    "Annual Income (k$)" in result.columns
    and "Spending Score (1-100)" in result.columns
):

    fig, ax = plt.subplots(figsize=(9, 6))

    for cluster_id in sorted(result["Cluster"].unique()):

        cluster_data = result[
            result["Cluster"] == cluster_id
        ]

        ax.scatter(
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            label=f"Cluster {cluster_id}"
        )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")
    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

else:
    st.info(
        "Select Annual Income and Spending Score "
        "to see the customer segmentation chart."
    )


# ---------------------------------------------------------
# 14. Cluster profile
# ---------------------------------------------------------
st.header("🔟 Cluster Profile")

profile = (
    result
    .groupby("Cluster")[selected_features]
    .mean()
    .round(2)
)

st.write(
    "These averages help us understand what each cluster represents."
)

st.dataframe(
    profile,
    use_container_width=True
)


# ---------------------------------------------------------
# 15. Test New Customer Data
# ---------------------------------------------------------
st.header("1️⃣1️⃣ Test New Customer Data")

st.write("""
You can enter a **new customer** here and check which existing
cluster they are most similar to.

### Important technical note
Agglomerative Hierarchical Clustering does not have a normal
`.predict()` method like many supervised ML models.

So, for a new customer, this mini project uses a practical
assignment method:

```text
New Customer
      ↓
StandardScaler using the OLD data
      ↓
Compare distance to each existing cluster center
      ↓
Nearest cluster
```

The cluster center is calculated from the already-created clusters.
This is an **approximate assignment**, not a new hierarchical
retraining step.
""")

new_values = {}

for feature in selected_features:
    # Use the original data median as a safe default value.
    default_value = float(X[feature].median())

    new_values[feature] = st.number_input(
        f"New {feature}",
        min_value=0.0,
        value=default_value,
        step=1.0
    )

if st.button("🔍 Find Customer Cluster"):

    # Convert the new customer's values into a DataFrame.
    new_customer = pd.DataFrame(
        [new_values],
        columns=selected_features
    )

    # IMPORTANT:
    # Do NOT fit a new scaler on the new customer.
    # We use the scaler learned from the original dataset.
    new_customer_scaled = scaler.transform(new_customer)

    # Calculate one center for every existing cluster.
    # These centers are calculated in SCALED feature space.
    cluster_centers = []

    for cluster_id in sorted(result["Cluster"].unique()):

        cluster_rows = X.loc[
            result["Cluster"] == cluster_id,
            selected_features
        ]

        # Convert the existing cluster's rows to scaled values.
        cluster_scaled = scaler.transform(cluster_rows)

        # Mean of all scaled points = practical cluster center.
        center = cluster_scaled.mean(axis=0)

        cluster_centers.append(center)

    cluster_centers = np.array(cluster_centers)

    # Calculate Euclidean distance from the new customer
    # to every existing cluster center.
    distances = np.sqrt(
        ((cluster_centers - new_customer_scaled[0]) ** 2).sum(axis=1)
    )

    # Find the nearest cluster.
    nearest_index = int(np.argmin(distances))
    predicted_cluster = sorted(
        result["Cluster"].unique()
    )[nearest_index]

    st.success(
        f"🎯 New customer belongs to **Cluster {predicted_cluster}** "
        f"(approximate assignment)."
    )

    # Show distances so students can understand WHY
    # that cluster was selected.
    distance_table = pd.DataFrame({
        "Cluster": sorted(result["Cluster"].unique()),
        "Distance from New Customer": distances.round(3)
    })

    distance_table = distance_table.sort_values(
        "Distance from New Customer"
    )

    st.subheader("📏 Distance from Each Cluster")

    st.dataframe(
        distance_table,
        use_container_width=True
    )

    st.info(
        "Smaller distance = more similar to that cluster. "
        "The nearest cluster is selected."
    )


# ---------------------------------------------------------
# 16. Simple business interpretation
# ---------------------------------------------------------
st.header("1️⃣2️⃣ Business Interpretation")

st.write("""
Hierarchical Clustering only gives us numbers such as:

```text
Cluster 0
Cluster 1
Cluster 2
...
```

It does NOT automatically know names such as:

```text
Premium Customer
Budget Customer
Frequent Spender
```

We interpret the clusters using their feature averages.

For example:

```text
High income + High spending
→ Possible premium/high-value customers

Low income + Low spending
→ Possible budget customers

High income + Low spending
→ Possible careful/low-engagement customers

Low income + High spending
→ Possible frequent-spending customers
```

These are business interpretations, not labels learned by the algorithm.
""")


# ---------------------------------------------------------
# 16. Download clustered data
# ---------------------------------------------------------
st.header("1️⃣2️⃣ Download Results")

csv_data = result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Clustered CSV",
    data=csv_data,
    file_name="hierarchical_customer_segments.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# 17. Final revision
# ---------------------------------------------------------
st.divider()

st.subheader("🧠 Remember")

st.code("""
Hierarchical Clustering

Data
 ↓
Features
 ↓
Scaling
 ↓
Distance
 ↓
Merge closest clusters
 ↓
Linkage
 ↓
Dendrogram
 ↓
Cut / Choose clusters
 ↓
Final Cluster Labels
""")

st.caption(
    "Mini Project: Hierarchical Clustering + Streamlit"
)
