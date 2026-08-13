# Import Streamlit to build the interactive web application.
import streamlit as st
# Import pandas for reading and manipulating the dataset.
import pandas as pd
# Import numpy for numerical calculations.
import numpy as np
# Import matplotlib for charts.
import matplotlib.pyplot as plt
# Import StandardScaler to put features on a comparable scale.
from sklearn.preprocessing import StandardScaler
# Import PCA for dimensionality reduction.
from sklearn.decomposition import PCA
# Import train_test_split for optional evaluation.
from sklearn.model_selection import train_test_split
# Import RandomForestRegressor for an optional downstream supervised test.
from sklearn.ensemble import RandomForestRegressor
# Import regression metrics to evaluate the downstream model.
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure the Streamlit page.
st.set_page_config(page_title="PCA Student Analysis", page_icon="📊", layout="wide")

# Display the project title.
st.title("📊 PCA - Student Performance Analysis")
# Explain the main purpose of the project.
st.write("This project uses PCA to reduce many student features into fewer principal components.")
# Show the Kaggle source.
st.info("Dataset: Kaggle Student Performance and Learning Behavior Dataset.")
# Provide the Kaggle page for downloading the CSV.
st.markdown("[Open the Kaggle dataset](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style)")

# Define the expected dataset location.
DATA_PATH = "data/merged_dataset.csv"

# Try to load the Kaggle dataset.
try:
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(DATA_PATH)
# Show a friendly message if the CSV is missing.
except FileNotFoundError:
    # Tell the user exactly where the file should be placed.
    st.error("Dataset not found. Download `merged_dataset.csv` from Kaggle and put it inside the `data` folder.")
    # Stop the application until the dataset is available.
    st.stop()

# Display the basic dataset information.
st.subheader("1. Dataset Overview")
# Show the number of rows and columns.
st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
# Show the first five rows.
st.dataframe(df.head())

# Select numeric columns because PCA requires numerical input.
numeric_df = df.select_dtypes(include="number").copy()

# Detect columns that look like IDs.
id_columns = [col for col in numeric_df.columns if "id" in col.lower()]
# Remove ID columns because an ID is not a meaningful measurement.
numeric_df = numeric_df.drop(columns=id_columns, errors="ignore")

# Detect common target columns that we do not want to use as PCA input.
target_columns = [col for col in ["ExamScore", "FinalGrade"] if col in numeric_df.columns]
# Keep the target values separately for optional supervised evaluation.
target_df = df[target_columns].copy() if target_columns else pd.DataFrame()

# Remove target columns from PCA features to avoid target leakage.
pca_features = numeric_df.drop(columns=target_columns, errors="ignore").copy()

# Make sure at least two numeric features are available.
if pca_features.shape[1] < 2:
    st.error("At least two numeric feature columns are required for PCA.")
    st.stop()

# Fill missing values using the median of each feature.
pca_features = pca_features.fillna(pca_features.median())

# Display the features going into PCA.
st.subheader("2. Features Used for PCA")
st.write("Target columns are excluded from PCA when they exist.")
st.write(list(pca_features.columns))

# Explain the component selection.
st.subheader("3. Choose Number of Components")
# Calculate the maximum legal number of components.
max_components = min(pca_features.shape[0], pca_features.shape[1])
# Start with two components when possible.
default_components = min(2, max_components)
# Let the user choose how many components PCA should create.
n_components = st.slider("How many principal components do you want?", 1, max_components, default_components)

# Standardize the features before PCA.
scaler = StandardScaler()
# Learn the scaling values and transform the dataset.
X_scaled = scaler.fit_transform(pca_features)

# Create the PCA model.
pca = PCA(n_components=n_components)
# Fit PCA and transform the scaled dataset.
X_pca = pca.fit_transform(X_scaled)

# Create names such as PC1, PC2, PC3.
component_names = [f"PC{i+1}" for i in range(n_components)]
# Store the transformed data in a DataFrame.
pca_df = pd.DataFrame(X_pca, columns=component_names)

# Calculate the percentage of variance captured by each component.
explained = pca.explained_variance_ratio_
# Calculate the cumulative variance.
cumulative = np.cumsum(explained)

# Explain cumulative variance in beginner-friendly language.
st.subheader("4. Explained Variance & Cumulative Variance")
st.write("**Explained variance** tells us how much variation/information one component captures.")
st.write("**Cumulative variance** tells us how much variation is captured when we add components together.")

# Create the variance table.
variance_df = pd.DataFrame({
    "Component": component_names,
    "Explained Variance (%)": explained * 100,
    "Cumulative Variance (%)": cumulative * 100
})
# Display the variance table.
st.dataframe(variance_df)

# Show the total variance captured by the selected components.
st.metric("Total Variance Captured", f"{cumulative[-1] * 100:.2f}%")

# Create the explained/cumulative variance chart.
fig, ax = plt.subplots()
# Draw the individual explained variance bars.
ax.bar(component_names, explained * 100, alpha=0.7, label="Individual Variance")
# Draw the cumulative variance line.
ax.plot(component_names, cumulative * 100, marker="o", label="Cumulative Variance")
# Label the axes.
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance (%)")
# Add the chart title.
ax.set_title("Explained vs Cumulative Variance")
# Display the legend.
ax.legend()
# Display the chart.
st.pyplot(fig)

# Explain the difference with an example.
st.info("Example: If PC1 = 40% and PC2 = 25%, then PC1 + PC2 have a cumulative variance of 65%.")

# Show which original features contribute to each component.
st.subheader("5. Which Features Are Combined in PC1, PC2, ...?")
st.write("PCA creates each component by combining the original features. The **loading** shows how strongly each feature contributes to a component.")

# PCA components contain one weight for every original feature.
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_features.columns,
    columns=component_names
)
# Create an absolute-value table so the strongest contributions are easy to find.
abs_loadings = loadings.abs()

# Let the user select one component to inspect.
selected_component = st.selectbox("Select a component", component_names)
# Sort features by their absolute contribution to the selected component.
selected_loadings = pd.DataFrame({
    "Feature": loadings.index,
    "Loading": loadings[selected_component].values,
    "Absolute Contribution": abs_loadings[selected_component].values
}).sort_values("Absolute Contribution", ascending=False)

# Display the feature contribution table.
st.dataframe(selected_loadings)

# Show the top five feature contributors in plain language.
top_features = selected_loadings.head(5)["Feature"].tolist()
st.success(f"Top features contributing to {selected_component}: " + ", ".join(top_features))

# Display all component loadings.
with st.expander("Show all PCA loadings"):
    st.dataframe(loadings)

# Visualize PC1 and PC2 when possible.
if n_components >= 2:
    st.subheader("6. PCA 2D Visualization")
    # Create a new figure.
    fig2, ax2 = plt.subplots()
    # Plot PC1 against PC2.
    ax2.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
    # Label the axes.
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    # Add the title.
    ax2.set_title("Students After PCA")
    # Display the plot.
    st.pyplot(fig2)

# Explain new-data transformation.
st.subheader("7. Test With One New Student")
st.write("Enter a new student's values. PCA will transform that new student into PC1, PC2, etc. using the same scaler and PCA learned from the dataset.")

# Create a form so all new-student inputs are submitted together.
with st.form("new_student_form"):
    # Create a dictionary to store the new student's feature values.
    new_values = {}
    # Create an input for every PCA feature.
    for feature in pca_features.columns:
        # Get a sensible default value from the dataset median.
        default_value = float(pca_features[feature].median())
        # Show a numeric input for the feature.
        new_values[feature] = st.number_input(feature, value=default_value)
    # Create the submit button.
    submitted = st.form_submit_button("Transform New Student")

# Process the new student when the form is submitted.
if submitted:
    # Convert the new student's values into a one-row DataFrame.
    new_student = pd.DataFrame([new_values], columns=pca_features.columns)
    # Standardize the new student using the already-fitted scaler.
    new_scaled = scaler.transform(new_student)
    # Transform the new student using the already-fitted PCA.
    new_pca = pca.transform(new_scaled)
    # Create a readable DataFrame for the new student's components.
    new_pca_df = pd.DataFrame(new_pca, columns=component_names)
    # Display the transformed result.
    st.success("New student transformed successfully!")
    st.dataframe(new_pca_df)

# Show the transformed dataset.
st.subheader("8. Transformed Dataset")
st.write("These are the new PCA features created from the original features.")
st.dataframe(pca_df.head(20))

# Calculate reconstruction error.
st.subheader("9. PCA Evaluation - Reconstruction Error")
st.write("PCA does not have normal classification accuracy because PCA is not a prediction/classification algorithm.")
st.write("A useful PCA evaluation is **reconstruction error**: how much information is lost when we reduce the dimensions and reconstruct the original scaled data.")

# Reconstruct the scaled data from the selected components.
X_reconstructed = pca.inverse_transform(X_pca)
# Calculate mean squared reconstruction error.
reconstruction_mse = np.mean((X_scaled - X_reconstructed) ** 2)
# Display the reconstruction error.
st.metric("Reconstruction MSE", f"{reconstruction_mse:.6f}")
# Explain the interpretation.
st.caption("Lower reconstruction error generally means less information was lost during dimensionality reduction.")

# Show a simple downstream supervised evaluation when ExamScore exists.
if "ExamScore" in df.columns:
    st.subheader("10. Optional Evaluation - Does PCA Help a Prediction Model?")
    st.write("This is NOT PCA accuracy. Here we use PCA components as inputs to a separate regression model and predict ExamScore.")

    # Create a clean target series.
    evaluation_df = pd.concat([pca_features, df["ExamScore"]], axis=1).dropna()
    # Separate the PCA input features and target.
    X_eval = evaluation_df[pca_features.columns]
    y_eval = evaluation_df["ExamScore"]

    # Split the original features into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X_eval, y_eval, test_size=0.2, random_state=42
    )

    # Fit a scaler only on the training data to avoid data leakage.
    eval_scaler = StandardScaler()
    X_train_scaled = eval_scaler.fit_transform(X_train)
    X_test_scaled = eval_scaler.transform(X_test)

    # Fit PCA only on the training data.
    eval_pca = PCA(n_components=n_components)
    X_train_pca = eval_pca.fit_transform(X_train_scaled)
    X_test_pca = eval_pca.transform(X_test_scaled)

    # Create a Random Forest regression model.
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the regression model using PCA components.
    regressor.fit(X_train_pca, y_train)
    # Predict ExamScore for the test data.
    y_pred = regressor.predict(X_test_pca)

    # Calculate R-squared.
    r2 = r2_score(y_test, y_pred)
    # Calculate mean absolute error.
    mae = mean_absolute_error(y_test, y_pred)
    # Calculate root mean squared error.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Display the evaluation metrics.
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")

    # Explain the metrics.
    st.write("**R²:** closer to 1 is generally better. **MAE:** average absolute prediction error. **RMSE:** penalizes larger errors more strongly.")
    st.warning("This evaluates a Random Forest model using PCA features; it does not measure PCA itself.")

# Display the key learning takeaway.
st.subheader("11. Key Learning")
st.success("You choose HOW MANY components you want. PCA automatically decides HOW to combine the original features into those components.")
