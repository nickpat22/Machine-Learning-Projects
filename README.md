# California Housing Price Prediction

A Machine Learning regression project that predicts housing values using the California Housing dataset.

This project explores individual and ensemble regression algorithms to understand how different machine learning models perform on housing price prediction tasks.

## Features

- Loads the California Housing dataset using Pandas.
- Separates input features and target values.
- Splits the dataset into training and testing subsets.
- Implements Decision Tree Regression.
- Implements Bagging Regression using Decision Tree as the base estimator.
- Implements Gradient Boosting Regression.
- Trains models using Scikit-learn.
- Predicts housing values using trained regression models.
- Evaluates model performance using Mean Squared Error (MSE) and R² Score.

## Machine Learning Algorithms

### 1. Decision Tree Regression

Uses a Decision Tree Regressor to predict housing values based on input features.

### 2. Bagging Regression

Implements Bootstrap Aggregating (Bagging) using Decision Tree Regression as the base estimator with 10 estimators.

### 3. Gradient Boosting Regression

Uses Gradient Boosting Regression with the following configuration:

- Number of estimators: 100
- Learning rate: 0.1
- Maximum depth: 3

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-learn
- **Algorithms:** Decision Tree, Bagging, Gradient Boosting
- **Model Evaluation:** Mean Squared Error (MSE), R² Score

## Project Structure

```text
03_House_Price_Prediction/
│
├── README.md
├── requirements.txt
├── california_housing.csv
│
├── Indivdual_California_DT.py
├── Ensemble_California_Bagging.py
└── Ensemble_California_Boosting.py
```

## Prerequisites

- Python 3.x
- pip (Python Package Installer)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/nickpat22/Machine-Learning-Projects.git
```

**2. Navigate to the project directory**

```bash
cd Machine-Learning-Projects/03_House_Price_Prediction
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## How to Run

Execute the individual Python scripts to train and evaluate different regression models.

**1. Decision Tree Regression**

```bash
python Indivdual_California_DT.py
```

**2. Bagging Regression**

```bash
python Ensemble_California_Bagging.py
```

**3. Gradient Boosting Regression**

```bash
python Ensemble_California_Boosting.py
```

## Machine Learning Workflow

1. Load the California Housing dataset.
2. Display dataset dimensions and sample records.
3. Separate independent features and dependent target values.
4. Split the dataset into training and testing subsets using an 80:20 ratio.
5. Create the selected regression model.
6. Train the model using training data.
7. Generate predictions using testing data.
8. Evaluate model performance using MSE and R² Score.

## Model Evaluation

The project uses the following evaluation metrics:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Lower values indicate smaller prediction errors.
- **R² Score:** Measures how well the model explains variations in the target variable. Higher values generally indicate better model fit.

## Learning Outcomes

Through this project, I explored:

- Supervised Machine Learning.
- Regression problems and predictive modelling.
- Decision Tree Regression.
- Ensemble Learning.
- Bootstrap Aggregating (Bagging).
- Gradient Boosting.
- Train-Test Splitting.
- Model training and prediction.
- Regression evaluation metrics.
- Comparing individual and ensemble learning algorithms.

## Author

**Nikunj Patidar**

GitHub: [@nickpat22](https://github.com/nickpat22)

---

*This project is part of my Machine Learning learning journey, focused on regression algorithms, ensemble learning, and predictive analytics.*
