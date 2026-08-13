# PCA + Kaggle + Streamlit Mini Project (Updated)

## Dataset

This project uses the **Student Performance and Learning Behavior Dataset** from Kaggle.

Kaggle:
https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style

The Kaggle page describes 14,003 records and 16 attributes, including StudyHours, Attendance, Resources, Motivation, Age, OnlineCourses, Discussions, AssignmentCompletion, ExamScore, StressLevel, and FinalGrade.

## New Features Added

### 1. Explained Variance

**Explained variance** tells us how much variation/information is captured by one principal component.

Example:

```text
PC1 = 40%
PC2 = 25%
PC3 = 15%
```

### 2. Cumulative Variance

**Cumulative variance** adds the explained variance of the components together.

```text
PC1 = 40%
PC2 = 25%

Cumulative after PC2 = 40 + 25 = 65%
```

So PC1 + PC2 together capture 65% of the variation.

### 3. Which Features Are Combined?

PCA creates each component from combinations of the original features.

The app displays **PCA loadings**.

Example:

```text
Feature             PC1 Loading
StudyHours             0.55
Attendance             0.48
AssignmentCompletion   0.42
StressLevel            -0.10
```

A larger absolute loading means that feature has a stronger contribution to that component.

A negative loading is also meaningful: it means the feature contributes in the opposite direction to that component.

Important:

> PCA does not simply put selected columns into PC1 or PC2. Each component is a mathematical combination of many original features.

### 4. Test One New Student

The app has a **Test With One New Student** section.

Enter values for the original PCA features.

The app then:

```text
New Student
    ↓
Same StandardScaler
    ↓
Same PCA
    ↓
PC1, PC2, PC3...
```

This is important because new data must be transformed using the **same scaler and PCA learned from the original dataset**.

### 5. PCA Evaluation

PCA itself does not have normal classification accuracy.

Why?

Because PCA is not predicting:

```text
Pass / Fail
Spam / Not Spam
Price
```

PCA is transforming data.

Instead, this project calculates:

#### Reconstruction MSE

PCA reduces the dimensions and then reconstructs the data.

```text
Original data
     ↓
    PCA
     ↓
Reduced data
     ↓
Reconstruct
     ↓
Approximate original data
```

The reconstruction error tells us how much information was lost.

**Lower reconstruction error is generally better.**

### 6. Optional Prediction Evaluation

Because the dataset contains `ExamScore`, the app also demonstrates a separate supervised evaluation.

It uses:

```text
Original features
      ↓
StandardScaler
      ↓
PCA
      ↓
PC1, PC2...
      ↓
Random Forest Regressor
      ↓
ExamScore prediction
```

It reports:

- R² Score
- MAE
- RMSE

This is **not PCA accuracy**.

It answers a different question:

> "How well can a separate prediction model work when it uses PCA components?"

## Libraries

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

## Project Structure

```text
PCA_Kaggle_Streamlit_Mini_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── merged_dataset.csv
```

## Setup in VS Code

### 1. Download the dataset

Download `merged_dataset.csv` from Kaggle and put it here:

```text
data/merged_dataset.csv
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate on Windows

```bash
venv\Scripts\activate
```

### 4. Install libraries

```bash
pip install -r requirements.txt
```

### 5. Run

```bash
streamlit run app.py
```

## Main PCA Flow

```text
Kaggle Dataset
      ↓
Select Numeric Features
      ↓
Remove ID / Target Columns
      ↓
Handle Missing Values
      ↓
StandardScaler
      ↓
PCA
      ↓
PC1, PC2, PC3...
```

## Important Beginner Concept

Remember:

> **We decide HOW MANY components we want; PCA decides HOW to create them.**

For example:

```python
PCA(n_components=2)
```

means:

```text
Many original features
        ↓
       PCA
        ↓
PC1 + PC2
```

The app now lets you see exactly which original features contribute most strongly to PC1, PC2, etc.
