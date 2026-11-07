"""
Project Name : Iris Classification Case Study
Description  : Explores the Iris dataset provided by Scikit-learn.
               Displays the dataset object, feature names, target
               class names, metadata counts, and every sample with
               its corresponding target label.

Author       : Nikunj Patidar
Date         : 07/11/2026
"""

from sklearn.datasets import load_iris


def main():
    """
    Function Name : main
    Description  : Loads the Iris dataset and displays its contents,
                   feature and target metadata, and sample records
                   with their corresponding class labels.

    Input        : None
    Output       : Prints the Iris dataset, feature names, target
                   names, their counts, and all sample IDs with
                   feature values and target labels.
    """

    # Step 1: Display the case study heading.
    Border = "-" * 30

    print(Border)
    print("Iris Classification Case Study")
    print(Border)

    # Step 2: Load the built-in Iris dataset from Scikit-learn.
    Dataset = load_iris()

    # Step 3: Display the complete dataset object.
    # This includes the data, target labels, feature names,
    # target names, and other dataset information.
    print("\nComplete Iris Dataset:")
    print(Dataset)

    # Step 4: Display the independent variable (feature) names.
    print("\nIndependent Variables:")
    print(Dataset.feature_names)
    print("Length of Independent Variables:", len(Dataset.feature_names))

    # Step 5: Display the dependent variable (target class) names.
    print("\nDependent Variables:")
    print(Dataset.target_names)
    print("Length of Dependent Variables:", len(Dataset.target_names))

    # Step 6: Display every sample with its feature values and target label.
    # Each sample has an ID, four feature values, and a numeric class label.
    print("\nIris Dataset Records:")

    for i in range(len(Dataset.target)):
        print(
            "ID %d, Features %s, Label %s"
            % (i, Dataset.data[i], Dataset.target[i])
        )


if __name__ == "__main__":
    main()
