################################################################
#
# Project Name : Advertising Sales Prediction
# Description  : Predicts Sales using Linear Regression
# Language     : Python
# Author       : Nikunj Patidar
#
################################################################


################################################################
#
# Importing Required Libraries
#
################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


################################################################
#
# Function Name : AdvertisingRegression
# Input         : Name of CSV file
# Output        : Performs complete Linear Regression workflow
# Description   : Loads the Advertising dataset, performs data
#                 analysis, trains a Linear Regression model,
#                 evaluates predictions and displays coefficients.
# Author        : Nikunj Patidar
#
################################################################

def AdvertisingRegression(DataPath):

    border = "-" * 40

    # Step 1: Load the Data
    print(border)
    print("Step 1: Load the Data")
    print(border)

    df = pd.read_csv(DataPath)

    print(df.head())


    # Step 2: Remove Unwanted Columns
    print(border)
    print("Step 2: Remove Unwanted Columns")
    print(border)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.head())


    # Step 3: Check Missing Values
    print(border)
    print("Step 3: Check Missing Values")
    print(border)

    print("Total Missing Values : ")
    print(df.isnull().sum())
    print(border)


    # Step 4: Statistical Summary
    print(border)
    print("Step 4: Statistical Summary")
    print(border)

    print(df.describe())


    # Step 5: Correlation
    print(border)
    print("Step 5: Correlation")
    print(border)

    print(df.corr())


    # Step 6: Separate Independent and Dependent Variables
    print(border)
    print("Step 6: Separate Independent and Dependent Variables")
    print(border)

    X = df[["TV", "radio", "newspaper"]]
    Y = df["sales"]

    print("Independent Variables :")
    print(X.head())

    print("Dependent Variables :")
    print(Y.head())


    # Step 7: Split the Dataset
    print(border)
    print("Step 7: Split the Dataset")
    print(border)

    x_train, x_test, y_tarin, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Training Data : ", x_train.shape)
    print("Testing Data : ", x_test.shape)


    # Step 8: Create and Train the Model
    print(border)
    print("Step 8: Create and Train the Model")
    print(border)

    model = LinearRegression()

    model = model.fit(x_train, y_tarin)

    print("Model Trained Successfully")


    # Step 9: Test the Model
    print(border)
    print("Step 9: Test the Model")
    print(border)

    y_pred = model.predict(x_test)

    print("Expected Answers : ")
    print(y_test[:3])

    print("Predicted Answers : ")
    print(y_pred[:3])


    # Step 10: Evaluate the Model
    print(border)
    print("Step 10: Evaluate the Model")
    print(border)

    MSE = mean_squared_error(y_test, y_pred)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(y_test, y_pred)

    print("MSE : ", MSE)
    print("RMSE : ", RMSE)
    print("R2 : ", R2)


    # Step 11: Display Coefficients
    print(border)
    print("Step 11: Display Coefficients")
    print(border)

    print("TV Coefficient : ", model.coef_[0])

    print("Radio Coefficient : ", model.coef_[1])

    print("Newspaper Coefficient : ", model.coef_[2])

    print("Intercept : ", model.intercept_)


################################################################
#
# Function Name : main
# Input         : None
# Output        : Executes the Advertising Regression program
# Description   : Calls the AdvertisingRegression function
#                 and performs the complete model workflow.
# Author        : Nikunj Patidar
#
################################################################

def main():

    # Execute the Advertising Regression program
    AdvertisingRegression("Advertising.csv")


################################################################
#
# Starter of the Program
#
################################################################

if __name__ == "__main__":
    main()
