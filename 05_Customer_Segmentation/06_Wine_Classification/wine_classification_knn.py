"""
Project Name : Wine Classification using K-Nearest Neighbors (KNN)
Description  : Classifies wine samples using the K-Nearest Neighbors
               algorithm. The dataset is cleaned, input features are
               standardized, and model accuracy is evaluated for
               K values from 1 to 20.

Author       : Nikunj Patidar
Date         : 07/11/2026
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def WineClassifier(DataPath):
    """
    Function Name : WineClassifier
    Description  : Loads and cleans the wine dataset, separates input
                   features and target labels, splits the data, scales
                   the features, and evaluates KNN models for different
                   values of K. Displays the accuracy results and graph.

    Input        : DataPath - path of the wine dataset CSV file.
    Output       : Displays dataset details, training/testing shapes,
                   accuracy scores for K values from 1 to 20, and an
                   accuracy-versus-K graph.
    """

    Border = "-" * 40

    # Step 1: Load the dataset from the CSV file.
    print(Border)
    print("Step 1: Load the Dataset")
    print(Border)

    df = pd.read_csv(DataPath)

    print("Some Entries from Dataset:")
    print(df.head())
    print(Border)

    # Step 2: Clean the dataset by removing rows containing missing values.
    print("Step 2: Clean the Dataset")
    print(Border)

    df.dropna(inplace=True)

    print("Shape of Dataset:", df.shape)
    print("Total Records:", df.shape[0])
    print("Total Columns:", df.shape[1])
    print(Border)

    # Step 3: Separate independent variables (X) and dependent variable (Y).
    # All columns except 'class_label' are used as input features.
    X = df.drop(columns=["class_label"])
    Y = df["class_label"]

    print("Step 3: Separate Input and Output Variables")
    print(Border)
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    print("Input Columns:", X.columns.tolist())
    print("Output Column: class_label")
    print(Border)

    # Step 4: Split the dataset into training and testing sets.
    # stratify=Y preserves the class distribution in both sets.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Step 4: Split the Dataset")
    print(Border)
    print("Details of Training and Testing Data:")
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of Y_train:", Y_train.shape)
    print("Shape of Y_test:", Y_test.shape)
    print(Border)

    # Step 5: Standardize the input features.
    # Fit the scaler only on training data to prevent data leakage.
    # Apply the same learned scaling parameters to the test data.
    print("Step 5: Feature Scaling")
    print(Border)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed.")
    print(Border)

    # Step 6: Evaluate KNN models for different values of K.
    # Accuracy is calculated for each K value from 1 to 20.
    accuracy_scores = []
    K_values = range(1, 21)

    print("Step 6: Evaluate Different K Values")
    print(Border)

    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, Y_train)

        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)

        accuracy_scores.append(accuracy)

    # Display the accuracy corresponding to each K value.
    print("Accuracy Report:")
    for k, accuracy in zip(K_values, accuracy_scores):
        print(f"K = {k}: Accuracy = {accuracy:.4f}")

    print(Border)

    # Step 7: Plot the relationship between K values and accuracy.
    # This graph helps visualize how the selected K value affects accuracy.
    print("Step 7: Graphical Representation")
    print(Border)

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, accuracy_scores, marker="o")
    plt.title("K Values vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.tight_layout()
    plt.show()


def main():
    """
    Function Name : main
    Description  : Entry point of the program. Calls the wine
                   classification function with the dataset path.

    Input        : None
    Output       : Starts the wine classification workflow.
    """

    WineClassifier("wine_data.csv")


if __name__ == "__main__":
    main()
