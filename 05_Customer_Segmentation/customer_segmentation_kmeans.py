"""
Project Name : Customer Segmentation using K-Means Clustering
Description  : Applies K-Means clustering to customer data using
               Annual Income and Spending Score. Uses the Elbow
               Method to inspect WCSS values for different cluster
               counts, then assigns customers to four clusters.

Author       : Nikunj Patidar
Date         : 07/11/2026
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    """
    Function Name : main
    Description  : Loads the Mall Customers dataset, selects the
                   required features, scales the data, performs
                   Elbow Method analysis, and assigns customers
                   to clusters using K-Means.

    Input        : Mall_Customers.csv
    Output       : Displays dataset information, missing-value
                   counts, selected features, scaled values,
                   WCSS values, the Elbow Method graph, and
                   the first 20 rows with cluster labels.
    """

    # Step 1: Load the dataset.
    df = pd.read_csv("Mall_Customers.csv")

    print("Dataset loaded successfully.")
    print(df.head())

    # Check and display the number of missing values in each column.
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Step 2: Select the features used for customer segmentation.
    # Annual Income and Spending Score are the two input features.
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    print("\nSelected Features:")
    print(X.head())

    # Step 3: Scale the selected features.
    # StandardScaler standardizes each feature before clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nScaled Data:")
    print(X_scaled[:5])

    # Step 4: Apply the Elbow Method.
    # WCSS (Within-Cluster Sum of Squares) is recorded for
    # cluster counts ranging from 1 to 10.
    WCSS = []

    for k in range(1, 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    # Display the WCSS value for each tested cluster count.
    print("\nWCSS Values:")
    for i in range(len(WCSS)):
        print(f"{i + 1}: {WCSS[i]}")

    # Step 5: Visualize the Elbow Method results.
    # The bend in the curve can help estimate a suitable
    # number of clusters.
    plt.plot(range(1, 11), WCSS, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method Analysis")
    plt.grid(True)
    plt.show()

    # Step 6: Build the final K-Means model.
    # The original implementation uses four clusters.
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    # Fit the model and obtain a cluster label for each customer.
    clusters = model.fit_predict(X_scaled)

    # Add the assigned cluster label to the original dataset.
    df["Cluster"] = clusters

    print("\nDataset with Cluster Labels:")
    print(df.head(20))


if __name__ == "__main__":
    main()
