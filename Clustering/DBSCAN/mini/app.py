# ============================================================
# DBSCAN Mini Project — Two Moons Dataset
# ============================================================

# Streamlit → creates the web application
import streamlit as st

# Pandas → works with table/data
import pandas as pd

# NumPy → numerical calculations
import numpy as np

# Matplotlib → creates graphs
import matplotlib.pyplot as plt

# make_moons → creates a simple curved dataset
from sklearn.datasets import make_moons

# DBSCAN → clustering algorithm
from sklearn.cluster import DBSCAN

# StandardScaler → scales features
from sklearn.preprocessing import StandardScaler

# Silhouette Score → evaluates cluster separation
from sklearn.metrics import silhouette_score


# ============================================================
# 1. Streamlit Page Settings
# ============================================================

st.set_page_config(
    page_title="DBSCAN - Two Moons",
    page_icon="🟢",
    layout="wide"
)

st.title("🟢 DBSCAN — Two Moons Clustering")

st.write(
    "A beginner-friendly Unsupervised Machine Learning "
    "project using DBSCAN."
)


# ============================================================
# 2. Explain DBSCAN
# ============================================================

with st.expander("🧠 What are we doing in this project?"):

    st.markdown("""
### DBSCAN

DBSCAN is a **density-based clustering algorithm**.

It finds:

- Dense areas → Clusters
- Isolated points → Noise

### Important parameters

```text
eps
↓
How far should we look?

min_samples
↓
How many nearby points are required?
```

### Three important terms

```text
Core Point
→ Enough nearby points

Border Point
→ Near a Core Point

Noise
→ Not connected to a dense region
```

### Important

This is **Unsupervised Learning**.

```text
Features → YES ✅
Target   → NO ❌
```
""")


# ============================================================
# 3. Create Two Moons Dataset
# ============================================================

st.header("1️⃣ Create Dataset")

n_samples = st.slider(
    "Number of data points",
    min_value=100,
    max_value=1000,
    value=500,
    step=50
)

noise = st.slider(
    "Dataset noise",
    min_value=0.00,
    max_value=0.30,
    value=0.05,
    step=0.01
)

# Two Moons creates two curved groups.
X, original_labels = make_moons(
    n_samples=n_samples,
    noise=noise,
    random_state=42
)


# ============================================================
# 4. Show Original Dataset
# ============================================================

st.subheader("Original Data")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    X[:, 0],
    X[:, 1],
    s=40
)

ax.set_title("Two Moons Dataset")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

st.pyplot(fig)

plt.close(fig)

st.info("""
This dataset has only **2 features**:

```text
Feature 1
Feature 2
```

We use a simple dataset so we can easily visualize what DBSCAN
is doing.
""")


# ============================================================
# 5. Feature Scaling
# ============================================================

st.header("2️⃣ Feature Scaling")

st.write("""
DBSCAN uses distance, so we normally scale the features before
clustering.
""")

scaler = StandardScaler()

# Learn scaling values and transform the data.
X_scaled = scaler.fit_transform(X)


# ============================================================
# 6. DBSCAN Parameters
# ============================================================

st.header("3️⃣ DBSCAN Parameters")

col1, col2 = st.columns(2)

with col1:

    # eps = neighborhood radius
    eps = st.slider(
        "eps — Neighborhood Radius",
        min_value=0.05,
        max_value=1.00,
        value=0.20,
        step=0.01
    )

with col2:

    # min_samples = minimum nearby points
    min_samples = st.slider(
        "min_samples — Minimum Points",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )

st.write(
    f"Current settings: **eps = {eps}**, "
    f"**min_samples = {min_samples}**"
)


# ============================================================
# 7. Create and Run DBSCAN
# ============================================================

model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

# fit_predict learns the density structure and assigns labels.
# Noise is represented by -1.
labels = model.fit_predict(X_scaled)


# ============================================================
# 8. Cluster Information
# ============================================================

unique_labels = sorted(set(labels))

cluster_labels = [
    label
    for label in unique_labels
    if label != -1
]

number_of_clusters = len(cluster_labels)

number_of_noise = np.sum(labels == -1)


# ============================================================
# 9. Results Metrics
# ============================================================

st.header("4️⃣ DBSCAN Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clusters", number_of_clusters)

with col2:
    st.metric("Noise Points", int(number_of_noise))

with col3:
    st.metric("Total Points", len(X))


# ============================================================
# 10. Cluster Visualization
# ============================================================

st.header("5️⃣ Cluster Visualization")

fig, ax = plt.subplots(figsize=(9, 6))

for cluster_id in unique_labels:

    cluster_points = X[labels == cluster_id]

    if cluster_id == -1:

        # Noise is displayed using an X marker.
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            marker="x",
            s=100,
            label="Noise (-1)"
        )

    else:

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=50,
            label=f"Cluster {cluster_id}"
        )

ax.set_title("DBSCAN Clustering")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 11. Explain Result
# ============================================================

st.subheader("🧠 Understand the Graph")

st.write("""
DBSCAN searches for areas where points are close together.

```text
Dense Area
    ↓
Cluster

Isolated Point
    ↓
Noise
```

Noise points are represented by:

```text
-1
```
""")


# ============================================================
# 12. Core, Border and Noise Points
# ============================================================

st.header("6️⃣ Core, Border and Noise Points")

# DBSCAN stores indexes of Core Points.
core_indices = model.core_sample_indices_

# Create a Boolean mask for Core Points.
core_mask = np.zeros(
    len(X),
    dtype=bool
)

core_mask[core_indices] = True

core_count = np.sum(core_mask)

# A non-core point assigned to a real cluster is treated as
# a Border Point in this educational visualization.
border_mask = (
    (labels != -1)
    & (~core_mask)
)

border_count = np.sum(border_mask)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Core Points", int(core_count))

with col2:
    st.metric("Border Points", int(border_count))

with col3:
    st.metric("Noise", int(number_of_noise))


# ============================================================
# 13. Silhouette Score
# ============================================================

st.header("7️⃣ Silhouette Score")

# Remove noise before calculating the score.
non_noise_mask = labels != -1

if (
    number_of_clusters >= 2
    and np.sum(non_noise_mask) > number_of_clusters
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
### Easy Meaning

```text
Closer to +1
↓
Better separated clusters

Around 0
↓
Clusters overlap

Below 0
↓
Possible poor clustering
""")

else:

    st.warning(
        "Silhouette Score cannot be calculated with the "
        "current DBSCAN settings."
    )


# ============================================================
# 14. Show Cluster Labels
# ============================================================

st.header("8️⃣ Data with Cluster Labels")

result = pd.DataFrame({
    "Feature_1": X[:, 0],
    "Feature_2": X[:, 1],
    "Cluster": labels
})

st.dataframe(
    result.head(20),
    use_container_width=True
)


# ============================================================
# 15. Download Results
# ============================================================

st.header("9️⃣ Download Results")

csv_data = result.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Clustered Data",
    data=csv_data,
    file_name="dbscan_two_moons.csv",
    mime="text/csv"
)


# ============================================================
# 16. Test a New Data Point
# ============================================================

st.header("🔟 Test a New Data Point")

st.warning("""
DBSCAN does not have a normal `.predict()` method in
scikit-learn.

So this section uses a simple educational approach:

```text
New Point
    ↓
Check distance from Core Points
    ↓
Within eps?
    ↓
Yes → Assign nearest cluster
No  → Noise / Unknown
```
""")

new_feature_1 = st.number_input(
    "New Feature 1",
    value=0.0,
    step=0.1
)

new_feature_2 = st.number_input(
    "New Feature 2",
    value=0.0,
    step=0.1
)

if st.button("🔍 Check New Point"):

    # Convert user input into an array.
    new_point = np.array([
        [new_feature_1, new_feature_2]
    ])

    # Use the SAME scaler learned from the original data.
    new_point_scaled = scaler.transform(new_point)

    if len(core_indices) == 0:

        st.error(
            "There are no Core Points with the current "
            "DBSCAN settings. Try changing eps or min_samples."
        )

    else:

        # Get all Core Points.
        core_points = X_scaled[core_indices]

        # Calculate Euclidean distance from the new point
        # to every Core Point.
        distances = np.sqrt(
            np.sum(
                (
                    core_points
                    - new_point_scaled
                ) ** 2,
                axis=1
            )
        )

        # Find the nearest Core Point.
        nearest_index = np.argmin(distances)

        nearest_distance = distances[nearest_index]

        # Get original index and cluster of nearest Core Point.
        nearest_core_original_index = (
            core_indices[nearest_index]
        )

        nearest_cluster = labels[
            nearest_core_original_index
        ]

        if nearest_distance <= eps:

            st.success(
                f"🎯 New point is approximately connected "
                f"to **Cluster {nearest_cluster}**."
            )

            st.write(
                f"Distance from nearest Core Point: "
                f"**{nearest_distance:.3f}**"
            )

        else:

            st.warning(
                "⚠️ This point is outside the eps neighborhood "
                "of existing Core Points."
            )

            st.write(
                "Possible result: **Noise / Unknown**"
            )


# ============================================================
# 17. Final Revision
# ============================================================

st.divider()

st.header("🧠 Final DBSCAN Revision")

st.code("""
DBSCAN

Data
 ↓
StandardScaler
 ↓
eps
 ↓
min_samples
 ↓
Find Neighbors
 ↓
Enough neighbors?
 ↓
Core Point
 ↓
Expand Cluster
 ↓
Border Points
 ↓
Isolated Points
 ↓
Noise
""")

st.success("""
Remember:

K-Means      → Centroid Based

Hierarchical → Tree Based

DBSCAN       → Density Based
""")
