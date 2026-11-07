# Wine Classification using K-Nearest Neighbors (KNN)

## Project Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm to classify wine samples using their input features.

The project follows a machine learning workflow that includes dataset loading, data cleaning, feature selection, train-test splitting, feature scaling, model evaluation, and graphical analysis.

It evaluates different K values ranging from 1 to 20 and visualizes their corresponding accuracy scores.

## Features

- Load the wine dataset from a CSV file.
- Display the first five dataset records.
- Remove rows containing missing values.
- Separate independent and dependent variables.
- Use `class_label` as the target variable.
- Split the dataset into training and testing sets (80:20).
- Preserve class distribution using stratified splitting.
- Standardize features using `StandardScaler`.
- Train KNN classifiers for K values from 1 to 20.
- Calculate accuracy scores for different K values.
- Display the accuracy report.
- Visualize K values against accuracy using Matplotlib.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
06_Wine_Classification/
│
├── wine_classification_knn.py
├── wine_data.csv
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.x
- pip package manager

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nickpat22/Machine-Learning-Projects.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Machine-Learning-Projects/06_Wine_Classification
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Execute the Python file:

```bash
python wine_classification_knn.py
```

Make sure `wine_data.csv` is present in the same directory as the Python file.

## Algorithm Used

**K-Nearest Neighbors (KNN)**

KNN is a supervised machine learning algorithm used for classification. It predicts the class of a sample based on the classes of its nearest neighboring data points.

This project evaluates K values from 1 to 20 to compare their classification accuracy.

## Model Evaluation

The model uses **Accuracy Score** to evaluate classification performance for each tested K value.

A graph is generated to visualize the relationship between the number of neighbors (K) and model accuracy.

## Learning Outcomes

- Understanding supervised machine learning.
- Implementing the KNN classification algorithm.
- Understanding feature scaling.
- Performing stratified train-test splitting.
- Evaluating classification accuracy.
- Comparing different K values.
- Visualizing model evaluation results.

## Author

**Nikunj Patidar**
