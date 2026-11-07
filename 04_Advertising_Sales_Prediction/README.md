# Advertising Sales Prediction

## Project Overview

This project implements a **Linear Regression model** to predict product sales based on advertising expenditure across three platforms: TV, Radio, and Newspaper.

The project demonstrates the complete machine learning workflow, including data loading, preprocessing, exploratory data analysis, model training, prediction, and evaluation.

## Features

- Load the advertising dataset using Pandas.
- Remove the unwanted `Unnamed: 0` column if present.
- Check for missing values.
- Generate statistical summaries using `describe()`.
- Calculate the correlation matrix.
- Separate independent variables (`TV`, `radio`, `newspaper`) and the target variable (`sales`).
- Split the dataset into training and testing sets.
- Train a Linear Regression model.
- Predict sales using test data.
- Evaluate model performance using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- Display model coefficients and intercept.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
04_Advertising_Sales_Prediction/
│
├── advertising_sales_prediction.py
├── Advertising.csv
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
   cd Machine-Learning-Projects/04_Advertising_Sales_Prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Execute the Python file:

```bash
python advertising_sales_prediction.py
```

Make sure `Advertising.csv` is present in the same directory as the Python file.

## Model Evaluation

The model evaluates its predictions using MSE, RMSE, and R² score to measure prediction error and model performance.

## Learning Outcomes

- Understanding Linear Regression.
- Working with Pandas and NumPy.
- Performing basic exploratory data analysis.
- Splitting datasets into training and testing sets.
- Training and evaluating a regression model.
- Understanding regression coefficients and intercept.

## Author

**Nikunj Patidar**
