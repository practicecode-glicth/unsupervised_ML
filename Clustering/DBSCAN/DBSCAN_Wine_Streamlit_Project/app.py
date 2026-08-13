"""
DBSCAN Mini Project — Wine Dataset
==================================

Dataset:
    Wine dataset from scikit-learn.

Algorithm:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This is an UNSUPERVISED learning project:
    Features -> Required
    Target   -> Not used for clustering

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import DBSCAN
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================================================
# 1. Page settings
# =========================================================
st.set_page_config(
    page_title="DBSCAN Wine Clustering",
    page_icon="🟢",
    layout="wide"
)

st.title("🟢 DBSCAN — Wine Dataset Clustering")
st.write(
    "A beginner-friendly Unsupervised ML mini project "
    "using DBSCAN and Streamlit."
)


# =========================================================
# 2. Project explanation
# =========================================================
with st.expander("🧠 Understand the complete project"):

    st.markdown("""
### Project Flow

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
PCA for 2D Visualization
     ↓
Silhouette Score
     ↓
Cluster Analysis
```

### Important

DBSCAN is **Unsupervised Learning**.

Therefore:

```text
X / Features → YES ✅
y / Target    → NO ❌
```

The Wine dataset contains a target/class in the original dataset,
but this project intentionally **does not use it for DBSCAN**.
""")


# =========================================================
# 3. Load Wine dataset
# =========================================================
st.header("1️⃣ Load Dataset")

wine = load_wine()

# Convert the sklearn dataset into a pandas DataFrame.
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

st.success(
    f"Wine dataset loaded: {df.shape[0]} rows × {df.shape[1]} features"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# 4. Feature selection
# =========================================================
st.header("2️⃣ Select Features")

st.write(
    "DBSCAN uses the selected features to find dense regions."
)

selected_features = st.multiselect(
    "Select features for clustering:",
    options=list(df.columns),
    default=list(df.columns)
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

X = df[selected_features].copy()


# =========================================================
# 5. Scaling
# =========================================================
st.header("3️⃣ Feature Scaling")

st.write("""
DBSCAN is distance-based.

For example, one feature might range from 0–5 while another
might range from 0–1000.

A large-scale feature can dominate the distance.

So we use **StandardScaler**.
""")

scaler = StandardScaler()

# fit_transform learns the mean/std from the current dataset
# and transforms every selected feature.
X_scaled = scaler.fit_transform(X)


# =========================================================
# 6. DBSCAN parameters
# =========================================================
st.header("4️⃣ DBSCAN Parameters")

col1, col2 = st.columns(2)

with col1:

    # eps = neighborhood radius.
    eps = st.slider(
        "eps — neighborhood radius",
        min_value=0.10,
        max_value=3.00,
        value=1.00,
        step=0.05
    )

with col2:

    # min_samples = minimum number of nearby samples needed
    # for a point to be considered a core point.
    min_samples = st.slider(
        "min_samples — minimum neighbors",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )


st.info(
    f"Current settings: eps = {eps}, min_samples = {min_samples}"
)


# =========================================================
# 7. Train DBSCAN
# =========================================================
st.header("5️⃣ Run DBSCAN")

model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

# fit_predict finds clusters and returns labels.
# Noise points are represented by -1.
labels = model.fit_predict(X_scaled)

result = X.copy()
result["Cluster"] = labels


# =========================================================
# 8. Basic results
# =========================================================
st.header("6️⃣ DBSCAN Results")

unique_labels = sorted(set(labels))

cluster_labels = [
    label for label in unique_labels
    if label != -1
]

noise_count = int(np.sum(labels == -1))
cluster_count = len(cluster_labels)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Clusters Found", cluster_count)

with c2:
    st.metric("Noise Points", noise_count)

with c3:
    st.metric("Total Data Points", len(labels))


st.write("### Cluster Labels")

st.write("""
```text
0, 1, 2, ... → clusters
-1            → noise
```
""")

st.dataframe(
    result.head(20),
    use_container_width=True
)


# =========================================================
# 9. Core / Border / Noise explanation
# =========================================================
st.header("7️⃣ Core, Border and Noise")

# DBSCAN exposes core_sample_indices_.
# These are the indexes of points identified as core samples.
core_indices = model.core_sample_indices_

core_mask = np.zeros(len(X), dtype=bool)
core_mask[core_indices] = True

core_count = int(core_mask.sum())

# A non-core point assigned to a real cluster is treated
# as a border point in this educational visualization.
border_count = int(
    np.sum((labels != -1) & (~core_mask))
)

bc1, bc2, bc3 = st.columns(3)

with bc1:
    st.metric("Core Points", core_count)

with bc2:
    st.metric("Border Points", border_count)

with bc3:
    st.metric("Noise Points", noise_count)

st.write("""
### Easy meaning

```text
Core   → Enough nearby points
Border → Near a dense/core region
Noise  → Not connected to a cluster
```
""")


# =========================================================
# 10. PCA visualization
# =========================================================
st.header("8️⃣ Cluster Visualization with PCA")

st.write("""
The Wine dataset has many features, so we cannot easily draw
all dimensions on a normal 2D chart.

PCA reduces the selected features to 2 dimensions only for
visualization.

**Important:** PCA is NOT the DBSCAN clustering algorithm here.
DBSCAN still runs on the scaled selected features.
""")

pca = PCA(n_components=2)

# PCA transforms the scaled features into two dimensions.
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": labels
})

fig, ax = plt.subplots(figsize=(10, 6))

for cluster_id in sorted(plot_df["Cluster"].unique()):

    cluster_data = plot_df[
        plot_df["Cluster"] == cluster_id
    ]

    # Use a different marker for noise.
    if cluster_id == -1:
        ax.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            marker="x",
            s=90,
            label="Noise (-1)"
        )
    else:
        ax.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            s=70,
            label=f"Cluster {cluster_id}"
        )

ax.set_title("DBSCAN Clusters — PCA 2D View")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 11. Silhouette Score
# =========================================================
st.header("9️⃣ Silhouette Score")

# Silhouette Score needs at least 2 real clusters.
# Noise points are removed for this educational evaluation.
non_noise_mask = labels != -1

if (
    cluster_count >= 2
    and np.sum(non_noise_mask) > cluster_count
):

    score = silhouette_score(
        X_scaled[non_noise_mask],
        labels[non_noise_mask]
    )

    st.metric(
        "Silhouette Score",
        f"{score:.3f}"
    )

    st.write("""
### Easy interpretation

```text
Closer to +1 → Better separated clusters
Around 0     → Overlapping clusters
Below 0      → Possible poor separation
```

The score is only one evaluation clue. It should not be the
only reason for choosing DBSCAN parameters.
""")

else:

    st.warning(
        "Silhouette Score cannot be calculated with the current "
        "DBSCAN result. Try changing eps or min_samples so that "
        "at least two real clusters are created."
    )


# =========================================================
# 12. Cluster distribution
# =========================================================
st.header("🔟 Cluster Distribution")

counts = result["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    counts.index.astype(str),
    counts.values
)

ax.set_xlabel("Cluster Label")
ax.set_ylabel("Number of Data Points")
ax.set_title("Number of Points in Each DBSCAN Cluster")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 13. Cluster profile
# =========================================================
st.header("1️⃣1️⃣ Cluster Profile")

profile = (
    result
    .groupby("Cluster")[selected_features]
    .mean()
    .round(2)
)

st.write(
    "Average feature values help us understand what each cluster contains."
)

st.dataframe(
    profile,
    use_container_width=True
)


# =========================================================
# 14. Download results
# =========================================================
st.header("1️⃣2️⃣ Download Results")

csv_data = result.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download DBSCAN Results",
    data=csv_data,
    file_name="dbscan_wine_clusters.csv",
    mime="text/csv"
)


# =========================================================
# 15. Test new data
# =========================================================
st.header("1️⃣3️⃣ Test New Data")

st.warning("""
### Important technical point

DBSCAN does not have a normal `.predict()` method for unseen
data in scikit-learn.

Therefore, this section is intentionally different from a
supervised ML prediction system.

For learning purposes, the app compares a new point with the
existing DBSCAN core samples and checks whether it is within
the `eps` neighborhood of any core sample.

If it is close enough to a core sample, we assign it to the
nearest connected cluster.

Otherwise, it is shown as **Noise / Unknown**.

This is an educational approximation, not a true DBSCAN
retraining/prediction API.
""")

new_values = {}

for feature in selected_features:

    # Median is used as a reasonable default input.
    default_value = float(X[feature].median())

    new_values[feature] = st.number_input(
        f"New {feature}",
        value=default_value,
        step=0.1
    )


if st.button("🔍 Check New Data"):

    # Convert user input into a DataFrame.
    new_df = pd.DataFrame(
        [new_values],
        columns=selected_features
    )

    # Use the SAME scaler learned from the original data.
    new_scaled = scaler.transform(new_df)

    if len(core_indices) == 0:

        st.error(
            "The current DBSCAN settings created no Core Points. "
            "Change eps or min_samples and try again."
        )

    else:

        # Get scaled values of all core points.
        core_points_scaled = X_scaled[core_indices]

        # Calculate Euclidean distance from the new point
        # to every existing core point.
        distances = np.sqrt(
            (
                core_points_scaled - new_scaled[0]
            ) ** 2
        ).sum(axis=1)

        # Find the closest core point.
        nearest_position = int(
            np.argmin(distances)
        )

        nearest_distance = float(
            distances[nearest_position]
        )

        nearest_core_index = int(
            core_indices[nearest_position]
        )

        nearest_cluster = int(
            labels[nearest_core_index]
        )

        if nearest_distance <= eps:

            st.success(
                f"🎯 New data is approximately connected to "
                f"**Cluster {nearest_cluster}**."
            )

            st.write(
                f"Nearest core-point distance: "
                f"**{nearest_distance:.3f}**"
            )

        else:

            st.warning(
                "⚠️ New data is outside the eps neighborhood "
                "of the existing core points."
            )

            st.write(
                f"Nearest core-point distance: "
                f"**{nearest_distance:.3f}**"
            )

            st.write(
                "Possible result: **Noise / Unknown**"
            )


# =========================================================
# 16. Final revision
# =========================================================
st.divider()

st.header("🧠 Final DBSCAN Revision")

st.code("""
DBSCAN

        Data
          ↓
    Select Features
          ↓
     StandardScaler
          ↓
      eps + min_samples
          ↓
     Check Neighbors
          ↓
    ┌─────┴─────┐
    ↓           ↓
Enough       Not enough
neighbors?   neighbors
    ↓           ↓
 Core        Border/Noise
    ↓
Expand Cluster
    ↓
Find Dense Regions
    ↓
Final Clusters + Noise
""")

st.success(
    "Remember: K-Means = Centroid, "
    "Hierarchical = Tree, DBSCAN = Density."
)
