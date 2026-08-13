# ============================================================
# GMM Mini Project — Customer Segmentation
# ============================================================
#
# GMM = Gaussian Mixture Model
#
# This project demonstrates:
# 1. Unsupervised Learning
# 2. Soft Clustering
# 3. Gaussian Mixture Model
# 4. n_components
# 5. Probability of cluster membership
# 6. StandardScaler
# 7. AIC and BIC
# 8. Streamlit visualization
# 9. Testing a new customer
#
# Dataset:
# data/customers.csv
#
# Features:
# Annual_Income
# Spending_Score
#
# IMPORTANT:
# There is NO target column.
# This is an unsupervised learning project.
# ============================================================


# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------

# Streamlit → creates the web application
import streamlit as st

# Pandas → reads and manages CSV/table data
import pandas as pd

# NumPy → numerical calculations
import numpy as np

# Matplotlib → creates charts
import matplotlib.pyplot as plt

# GMM → Gaussian Mixture Model
from sklearn.mixture import GaussianMixture

# StandardScaler → puts features on a comparable scale
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 2. Page settings
# ------------------------------------------------------------

st.set_page_config(
    page_title="GMM Customer Segmentation",
    page_icon="🟣",
    layout="wide"
)

st.title("🟣 GMM — Customer Segmentation")

st.write(
    "A beginner-friendly Unsupervised ML mini project "
    "using Gaussian Mixture Model (GMM)."
)


# ------------------------------------------------------------
# 3. Explain the project
# ------------------------------------------------------------

with st.expander("🧠 Understand the complete project"):

    st.markdown("""
### Project Flow

```text
Customer Dataset
      ↓
Select Features
      ↓
StandardScaler
      ↓
Gaussian Mixture Model
      ↓
Probability of Each Cluster
      ↓
Final Cluster
      ↓
Visualization
```

### GMM asks:

> "What is the probability that this customer belongs to each cluster?"

Example:

```text
Cluster 0 → 10%
Cluster 1 → 25%
Cluster 2 → 65%
```

So GMM is called **Soft Clustering**.

### Important

This project is **Unsupervised Learning**.

```text
Features → YES ✅
Target   → NO ❌
```
""")


# ------------------------------------------------------------
# 4. Load dataset
# ------------------------------------------------------------

st.header("1️⃣ Load Customer Dataset")

# Read the CSV file from the data folder.
df = pd.read_csv("data/customers.csv")

st.success(
    f"Dataset loaded: {df.shape[0]} customers"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ------------------------------------------------------------
# 5. Explain the features
# ------------------------------------------------------------

st.header("2️⃣ Understand the Features")

st.write("""
We have two simple features:

```text
Annual_Income
→ Customer's annual income

Spending_Score
→ Customer's spending score
```

We will use these two features to discover customer groups.
""")


# ------------------------------------------------------------
# 6. Select features
# ------------------------------------------------------------

feature_columns = [
    "Annual_Income",
    "Spending_Score"
]

X = df[feature_columns].copy()


# ------------------------------------------------------------
# 7. Show original data
# ------------------------------------------------------------

st.subheader("Original Customer Data")

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(
    X["Annual_Income"],
    X["Spending_Score"],
    s=45
)

ax.set_title("Customers Before GMM")
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")

st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# 8. Feature Scaling
# ------------------------------------------------------------

st.header("3️⃣ Feature Scaling")

st.write("""
Income and spending score have very different numerical ranges.

For example:

```text
Annual Income → 30,000
Spending Score → 50
```

So we use StandardScaler before GMM.
""")

scaler = StandardScaler()

# Learn mean/std and transform the features.
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 9. Choose number of components
# ------------------------------------------------------------

st.header("4️⃣ Choose GMM Components")

n_components = st.slider(
    "Number of Gaussian Components",
    min_value=1,
    max_value=6,
    value=3,
    step=1
)

st.info(
    f"Current n_components = {n_components}"
)


# ------------------------------------------------------------
# 10. Create GMM model
# ------------------------------------------------------------

model = GaussianMixture(
    n_components=n_components,
    random_state=42
)


# ------------------------------------------------------------
# 11. Train GMM
# ------------------------------------------------------------

# fit() learns the Gaussian distributions.
model.fit(X_scaled)


# ------------------------------------------------------------
# 12. Predict cluster
# ------------------------------------------------------------

# predict() gives the most likely cluster.
labels = model.predict(X_scaled)


# ------------------------------------------------------------
# 13. Get probabilities
# ------------------------------------------------------------

# predict_proba() gives the probability for every component.
probabilities = model.predict_proba(X_scaled)


# ------------------------------------------------------------
# 14. Create result DataFrame
# ------------------------------------------------------------

result = df.copy()

result["Cluster"] = labels


# Add a column showing the highest probability.
result["Max_Probability"] = probabilities.max(
    axis=1
).round(3)


# ------------------------------------------------------------
# 15. Show metrics
# ------------------------------------------------------------

st.header("5️⃣ GMM Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Components",
        n_components
    )

with col2:
    st.metric(
        "Customers",
        len(df)
    )

with col3:
    st.metric(
        "Average Confidence",
        f"{probabilities.max(axis=1).mean():.2f}"
    )


# ------------------------------------------------------------
# 16. Cluster visualization
# ------------------------------------------------------------

st.header("6️⃣ Customer Clusters")

fig, ax = plt.subplots(figsize=(10, 6))

for cluster_id in range(n_components):

    cluster_data = X[
        labels == cluster_id
    ]

    ax.scatter(
        cluster_data["Annual_Income"],
        cluster_data["Spending_Score"],
        s=55,
        label=f"Cluster {cluster_id}"
    )

ax.set_title("GMM Customer Segmentation")
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# 17. Show probability
# ------------------------------------------------------------

st.header("7️⃣ GMM Probability — Soft Clustering")

st.write("""
Unlike K-Means, GMM can tell us the probability of a customer
belonging to every cluster.
""")

probability_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Cluster_{i}_Probability"
        for i in range(n_components)
    ]
)

probability_result = pd.concat(
    [
        df,
        result[["Cluster"]],
        probability_df
    ],
    axis=1
)

st.dataframe(
    probability_result.head(20),
    use_container_width=True
)


# ------------------------------------------------------------
# 18. Explain one customer
# ------------------------------------------------------------

st.header("8️⃣ Understand One Customer")

customer_index = st.number_input(
    "Customer row number",
    min_value=1,
    max_value=len(df),
    value=1,
    step=1
)

selected_index = customer_index - 1

selected_customer = df.iloc[selected_index]

selected_probabilities = probabilities[
    selected_index
]

st.write("### Customer Information")

st.write(
    f"Annual Income: **₹{selected_customer['Annual_Income']:,}**"
)

st.write(
    f"Spending Score: **{selected_customer['Spending_Score']}**"
)

st.write("### Cluster Probabilities")

probability_table = pd.DataFrame({
    "Cluster": [
        f"Cluster {i}"
        for i in range(n_components)
    ],
    "Probability": selected_probabilities
})

st.dataframe(
    probability_table,
    use_container_width=True
)

most_likely_cluster = int(
    np.argmax(selected_probabilities)
)

highest_probability = float(
    selected_probabilities[most_likely_cluster]
)

st.success(
    f"Most likely cluster: **Cluster {most_likely_cluster}** "
    f"with probability **{highest_probability:.2%}**"
)


# ------------------------------------------------------------
# 19. AIC and BIC
# ------------------------------------------------------------

st.header("9️⃣ AIC and BIC")

st.write("""
AIC and BIC help us compare different numbers of Gaussian
components.

Generally:

```text
Lower AIC → Better
Lower BIC → Better
```

They should be used as model-selection clues, not as the only
decision.
""")

aic_values = []
bic_values = []
component_values = range(1, 7)

for k in component_values:

    test_model = GaussianMixture(
        n_components=k,
        random_state=42
    )

    test_model.fit(X_scaled)

    aic_values.append(
        test_model.aic(X_scaled)
    )

    bic_values.append(
        test_model.bic(X_scaled)
    )

comparison = pd.DataFrame({
    "Components": list(component_values),
    "AIC": aic_values,
    "BIC": bic_values
})

st.dataframe(
    comparison,
    use_container_width=True
)


# ------------------------------------------------------------
# 20. AIC/BIC graph
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    list(component_values),
    aic_values,
    marker="o",
    label="AIC"
)

ax.plot(
    list(component_values),
    bic_values,
    marker="o",
    label="BIC"
)

ax.set_title("Choosing GMM Components")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Score")
ax.legend()

st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# 21. Cluster summary
# ------------------------------------------------------------

st.header("🔟 Cluster Summary")

summary = (
    result
    .groupby("Cluster")[feature_columns]
    .mean()
    .round(2)
)

st.dataframe(
    summary,
    use_container_width=True
)


# ------------------------------------------------------------
# 22. Download results
# ------------------------------------------------------------

st.header("1️⃣1️⃣ Download Results")

csv_data = probability_result.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download GMM Results",
    data=csv_data,
    file_name="gmm_customer_segmentation.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# 23. Test a New Customer
# ------------------------------------------------------------

st.header("1️⃣2️⃣ Test a New Customer")

st.write("""
Enter a new customer's income and spending score.

GMM will calculate the probability of this customer belonging
to every Gaussian component.
""")

new_income = st.number_input(
    "New Customer Annual Income",
    min_value=10000.0,
    max_value=200000.0,
    value=60000.0,
    step=1000.0
)

new_spending = st.number_input(
    "New Customer Spending Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)


if st.button("🔍 Predict New Customer"):

    # Put new customer into the same feature order.
    new_customer = pd.DataFrame({
        "Annual_Income": [new_income],
        "Spending_Score": [new_spending]
    })

    # Use the SAME scaler that was fitted on training data.
    new_customer_scaled = scaler.transform(
        new_customer
    )

    # Get cluster probabilities.
    new_probabilities = model.predict_proba(
        new_customer_scaled
    )[0]

    # Find most likely cluster.
    new_cluster = int(
        np.argmax(new_probabilities)
    )

    st.success(
        f"Most likely cluster: **Cluster {new_cluster}**"
    )

    st.write("### Probabilities")

    new_probability_table = pd.DataFrame({
        "Cluster": [
            f"Cluster {i}"
            for i in range(n_components)
        ],
        "Probability": [
            f"{p:.2%}"
            for p in new_probabilities
        ]
    })

    st.dataframe(
        new_probability_table,
        use_container_width=True
    )


# ------------------------------------------------------------
# 24. Final Revision
# ------------------------------------------------------------

st.divider()

st.header("🧠 Final GMM Revision")

st.code("""
GMM

Data
 ↓
StandardScaler
 ↓
Choose n_components
 ↓
Gaussian Mixture Model
 ↓
EM Algorithm
 ↓
Calculate Probabilities
 ↓
Most Likely Cluster
 ↓
Soft Clustering
""")

st.success("""
Remember:

K-Means      → Center / Distance

Hierarchical → Tree

DBSCAN       → Density

PCA          → Reduce Dimensions

GMM          → Probability
""")


# ============================================================
# END OF PROJECT
# ============================================================
