# Titanic Survival Predictor

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster using passenger-related information.

This project implements Logistic Regression to classify passenger survival based on features such as passenger class, gender, age, family members, fare, and embarkation location.

## Features

- Loads the Titanic dataset using Pandas.
- Performs data preprocessing and cleaning.
- Handles missing values using median and mode imputation.
- Removes unnecessary dataset columns.
- Converts categorical features into numerical representations using One-Hot Encoding.
- Splits the dataset into training and testing subsets.
- Trains a Logistic Regression classification model.
- Evaluates model performance using accuracy score and confusion matrix.
- Saves the trained model using Joblib.
- Loads the saved model for passenger survival prediction.
- Accepts passenger information through command-line input.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Algorithm:** Logistic Regression
- **Model Evaluation:** Accuracy Score, Confusion Matrix
- **Model Serialization:** Joblib

## Project Structure

```text
01_Titanic_Survival_Predictor/
│
├── README.md
├── requirements.txt
├── titanic_dataset.csv
├── titanic_model.pkl
│
├── TitanicLogistic1_LoadData.py
├── TitanicLogistic2_DataPreprocessing.py
├── TitanicLogistic3_SplitDataset.py
├── TitanicLogistic4_TrainModel.py
├── TitanicLogistic5_preserveModel.py
│
└── ModelLoadTitenic.py
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
cd Machine-Learning-Projects/01_Titanic_Survival_Predictor
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## Project Execution

The project is divided into multiple implementation stages.

### Step 1: Load Dataset

```bash
python TitanicLogistic1_LoadData.py.py
```

Loads the Titanic dataset using Pandas.

### Step 2: Data Preprocessing

```bash
python TitanicLogistic2_DataPreprocessing.py
```

Performs:
- Removal of unnecessary columns.
- Missing value handling.
- Categorical feature encoding.

### Step 3: Split Dataset

```bash
python TitanicLogistic3_SplitDataset.py
```

Divides the dataset into training and testing subsets using an 80:20 split.

### Step 4: Train and Evaluate Model

```bash
python TitanicLogistic4_TrainModel.py
```

Trains a Logistic Regression model and evaluates its performance using accuracy score and confusion matrix.

### Step 5: Save Trained Model

```bash
python TitanicLogistic5_preserveModel.py
```

Trains, evaluates and saves the trained model in a `.pkl` file using Joblib.

### Step 6: Load Model and Predict

```bash
python ModelLoadTitenic.py
```

Loads the saved model and accepts passenger details through command-line input to predict survival.

## Machine Learning Workflow

1. Dataset Loading
2. Data Preprocessing
3. Missing Value Imputation
4. Categorical Feature Encoding
5. Feature and Target Separation
6. Train-Test Splitting
7. Logistic Regression Model Training
8. Model Evaluation
9. Model Saving
10. Model Loading and Prediction

## Model Evaluation

The implementation uses the following evaluation techniques:

- **Accuracy Score:** Measures the proportion of correctly classified passenger records.
- **Confusion Matrix:** Displays the classification results by comparing actual and predicted survival labels.

## Learning Outcomes

Through this project, I explored:

- Supervised Machine Learning.
- Binary Classification.
- Logistic Regression.
- Data preprocessing and cleaning.
- Missing value imputation.
- One-Hot Encoding.
- Train-Test Splitting.
- Model training and evaluation.
- Model serialization using Joblib.
- Loading trained models for predictions.

## Author

**Nikunj Patidar**

GitHub: [@nickpat22](https://github.com/nickpat22)

---

*This project is part of my Machine Learning learning journey, focused on understanding classification algorithms and implementing an end-to-end machine learning workflow.*
