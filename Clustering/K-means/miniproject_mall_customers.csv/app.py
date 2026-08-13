"""
APP.PY — Streamlit application.

Run:
    streamlit run app.py

train.py trains and saves the model.
app.py loads that saved model and uses it.

The app does not need a target/y column.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Mall Customer Segmentation")
st.write(
    "K-Means clustering project using Annual Income and Spending Score."
)


# ---------------------------------------------------------
# 1. Explain the project
# ---------------------------------------------------------
with st.expander("🧠 Understand the complete flow"):
    st.markdown("""
```text
Kaggle CSV
    ↓
data/Mall_Customers.csv
    ↓
train.py
    ↓
Select X features
    ↓
StandardScaler
    ↓
Elbow Method
    ↓
K-Means
    ↓
Save model + scaler
    ↓
models/kmeans_model.joblib
    ↓
app.py
    ↓
Load saved model
    ↓
New customer
    ↓
Predict cluster
```
""")


# ---------------------------------------------------------
# 2. Load saved model
# ---------------------------------------------------------
MODEL_PATH = "models/kmeans_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found.")
    st.info(
        "First put Mall_Customers.csv in data/ and run: python train.py"
    )
    st.stop()

# Joblib loads the model, scaler, features and K saved by train.py.
saved = joblib.load(MODEL_PATH)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
k = saved["k"]


# ---------------------------------------------------------
# 3. Model information
# ---------------------------------------------------------
st.header("1️⃣ Model Information")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Algorithm", "K-Means")

with c2:
    st.metric("Clusters (K)", k)

with c3:
    st.metric("Features", len(features))

st.write("Features used:")
for feature in features:
    st.write(f"• {feature}")


# ---------------------------------------------------------
# 4. Show training data
# ---------------------------------------------------------
DATA_PATH = "data/Mall_Customers.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    st.header("2️⃣ Training Data")
    st.write(
        "This CSV is the data used to train K-Means. "
        "It has features but does not need a target/y column."
    )
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Total rows: **{len(df)}**")
else:
    st.warning(
        "The original CSV is not present. The saved model can still "
        "make predictions because the trained model and scaler are saved."
    )


# ---------------------------------------------------------
# 5. Show centroids
# ---------------------------------------------------------
st.header("3️⃣ Cluster Centers")

# Model centers are scaled. Convert them back to original units
# so they are easier for humans to understand.
centers_original = scaler.inverse_transform(model.cluster_centers_)

centers_df = pd.DataFrame(
    centers_original,
    columns=features
)
centers_df.index.name = "Cluster"

st.dataframe(
    centers_df.round(2),
    use_container_width=True
)


# ---------------------------------------------------------
# 6. New customer
# ---------------------------------------------------------
st.header("4️⃣ Predict a New Customer")

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    value=60.0,
    step=1.0
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)


# ---------------------------------------------------------
# 7. Predict
# ---------------------------------------------------------
if st.button("🔍 Predict Customer Cluster"):

    # Keep exactly the same feature order as training.
    new_customer = np.array([[income, spending]])

    # IMPORTANT:
    # transform() uses the scaler learned during training.
    # Do NOT fit_transform() on the new customer.
    new_customer_scaled = scaler.transform(new_customer)

    # K-Means returns the nearest learned cluster.
    predicted_cluster = int(
        model.predict(new_customer_scaled)[0]
    )

    st.success(
        f"🎯 This customer belongs to **Cluster {predicted_cluster}**."
    )

    st.dataframe(
        pd.DataFrame({
            "Annual Income (k$)": [income],
            "Spending Score (1-100)": [spending],
            "Predicted Cluster": [predicted_cluster]
        }),
        use_container_width=True
    )


# ---------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------
st.header("5️⃣ How to Understand the Clusters")

st.warning("Cluster numbers do not have fixed meanings.")

st.write("""
K-Means only creates groups such as Cluster 0, Cluster 1, etc.

We interpret them by looking at the cluster center.

For example:

- High income + high spending → possible premium customers
- Low income + low spending → possible budget customers
- High income + low spending → possible careful spenders
- Low income + high spending → possible frequent spenders

These are business interpretations, not labels directly learned by K-Means.
""")

st.divider()
st.caption("Beginner K-Means + Streamlit Project")
