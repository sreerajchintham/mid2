Here's a comprehensive `README.md` for the `Mid2.ipynb` notebook.

```markdown
# Heart Disease Prediction using K-Means Clustering

## Overview

This Google Colab notebook, `Mid2.ipynb`, demonstrates a machine learning approach to predict heart disease using K-Means clustering. Although K-Means is primarily an unsupervised clustering algorithm, this notebook cleverly adapts it for a classification-like task by aligning the learned clusters with the true labels of the training data. The process involves data cleaning, preprocessing, exploratory data analysis, determining an optimal number of clusters (`k`), training the K-Means model, and finally, using the trained model to predict heart disease risk for new patient data.

## Key Features

*   **Data Loading & Initial Exploration**: Loads the `heart.csv` dataset and performs initial checks for missing values and data types.
*   **Data Preprocessing**:
    *   Handles missing values by dropping rows.
    *   Applies `LabelEncoder` to transform categorical (object type) features into numerical representations.
*   **Exploratory Data Analysis (EDA)**:
    *   Generates a `pairplot` to visualize relationships between features, colored by the `HeartDisease` target variable.
    *   Computes and visualizes the correlation matrix using a heatmap to understand feature interdependencies.
*   **Optimal K Determination for K-Means**:
    *   Iteratively trains K-Means models for a range of `k` values (2 to 29).
    *   Calculates the Within-Cluster Sum of Squares (WCSS) for the Elbow Method.
    *   Calculates an "accuracy" metric by aligning each cluster's predicted label to the mode of its true labels in the training set, effectively using K-Means for pseudo-classification.
    *   Visualizes both the Elbow curve and the Accuracy curve to aid in selecting the best `k`.
*   **Model Training**: Trains the final K-Means model with the chosen optimal `k` on the preprocessed training data.
*   **New Patient Prediction**: Demonstrates how to predict heart disease risk (0 or 1) for a hypothetical new patient based on their attributes by assigning them to a cluster and then inferring the cluster's most frequent heart disease status from the training data.

## Technologies and Libraries Used

*   **Python**: The core programming language.
*   **pandas**: For data manipulation and analysis.
*   **scikit-learn (`sklearn`)**:
    *   `LabelEncoder`: For transforming categorical features.
    *   `train_test_split`: For splitting data into training and testing sets.
    *   `KMeans`: For performing K-Means clustering.
    *   `accuracy_score`: For evaluating the pseudo-classification performance.
*   **seaborn**: For statistical data visualization (`pairplot`, `heatmap`).
*   **matplotlib.pyplot**: For creating static, interactive, and animated visualizations (plotting WCSS and accuracy curves).
*   **numpy**: For numerical operations, especially array manipulation.
*   **scipy.stats.mode**: Used to find the most frequent true label within a cluster for alignment.

## Main Sections and Steps

The notebook follows a standard machine learning workflow:

1.  **Setup and Imports**: Importing all necessary libraries.
2.  **Data Loading**: Reading `heart.csv` into a pandas DataFrame.
3.  **Initial Data Inspection**: `df.head()`, `df.isna().sum()`, `df.describe()`, `df.info()`.
4.  **Data Cleaning**: `df.dropna(inplace=True)` to handle missing values.
5.  **Feature Encoding**: Identifying and converting categorical columns to numerical using `LabelEncoder`.
6.  **Exploratory Data Analysis (EDA)**:
    *   `sns.pairplot()`
    *   Correlation matrix calculation (`df.corr()`)
    *   Correlation heatmap visualization (`sns.heatmap()`)
7.  **Data Splitting**: Separating features (X) from the target (y) and then splitting into training and testing sets (`train_test_split`).
8.  **K-Means Optimal K Determination**:
    *   Looping through `k` values (2-29).
    *   Fitting `KMeans` for each `k`.
    *   Calculating `wcss` (inertia) and "accuracy" for each `k` by aligning cluster labels with true `y_train` labels.
9.  **Visualization of Optimal K**: Plotting WCSS vs. k (Elbow curve) and Accuracy vs. k. The chosen `best_k` is highlighted (set to 21 in this notebook).
10. **Final K-Means Model Training**: Training `KMeans` with the `best_k` on the `X_train` data.
11. **New Patient Prediction Example**:
    *   Creating a sample new patient DataFrame.
    *   Predicting the cluster for the new patient.
    *   Inferring the heart disease risk (0 or 1) based on the most frequent class in that cluster from the training data.

## Key Insights and Results

*   The notebook effectively demonstrates a method to use K-Means clustering for a binary classification problem (predicting Heart Disease) by mapping clusters to class labels.
*   The optimal number of clusters (`k`) is determined not only by the Elbow method on WCSS but also by observing the "accuracy" of the cluster-to-label mapping, which peaks at certain `k` values.
*   The chosen `best_k` for the final model training is 21, based on the combined analysis of WCSS and accuracy plots.
*   The `pairplot` and correlation heatmap provide insights into feature distributions and their relationships with `HeartDisease`. For instance, `Oldpeak` and `ST_Slope` show notable correlations.
*   The final output for a new patient is a predicted cluster label and the inferred `HeartDisease` risk (0 for no disease, 1 for disease).

## How to Use/Run the Notebook

To run this notebook, follow these steps:

1.  **Open in Google Colab**: Click on "Open in Colab" or upload the `Mid2.ipynb` file to your Google Drive and open it with Google Colaboratory.
2.  **Dataset**: Ensure the `heart.csv` dataset is available in your Colab environment. By default, the notebook expects it at `/content/heart.csv`. You can:
    *   Upload `heart.csv` directly to the Colab session by clicking the folder icon on the left sidebar -> Upload.
    *   Mount your Google Drive and specify the path if the file is there.
3.  **Run Cells**: Execute each cell sequentially by pressing `Shift + Enter` or by going to `Runtime -> Run all`.
4.  **Explore Outputs**: Review the outputs of each cell, including data head, summaries, plots, and the final prediction.

Feel free to modify the `new_patient` data in Cell 25 to test predictions for different hypothetical patients.
```