# Customer Segmentation using K-Means Clustering

## Project Overview

This project implements **K-Means Clustering** to segment mall customers based on their annual income and spending score.

The project uses the Elbow Method to analyze different cluster counts and visualize the Within-Cluster Sum of Squares (WCSS). A final K-Means model with four clusters is then trained to assign cluster labels to customers.

## Features

- Load customer data from a CSV file.
- Display the first five records.
- Check for missing values.
- Select Annual Income and Spending Score as clustering features.
- Standardize selected features using `StandardScaler`.
- Apply the Elbow Method for cluster counts from 1 to 10.
- Calculate and display WCSS values.
- Visualize the Elbow Method using Matplotlib.
- Train a final K-Means model with four clusters.
- Assign cluster labels to customers.
- Display the first 20 records with their cluster labels.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
05_Customer_Segmentation/
│
├── customer_segmentation_kmeans.py
├── Mall_Customers.csv
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
   cd Machine-Learning-Projects/05_Customer_Segmentation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Execute the Python file:

```bash
python customer_segmentation_kmeans.py
```

Make sure `Mall_Customers.csv` is present in the same directory as the Python file.

## Algorithm Used

**K-Means Clustering**

K-Means is an unsupervised machine learning algorithm that groups data points into clusters based on their similarity.

**Elbow Method:** Used to visualize WCSS values for different numbers of clusters and help identify a suitable cluster count.

## Learning Outcomes

- Understanding unsupervised machine learning.
- Implementing K-Means Clustering.
- Understanding feature scaling using StandardScaler.
- Applying the Elbow Method.
- Understanding WCSS.
- Assigning cluster labels to data points.
- Visualizing clustering analysis.

## Author

**Nikunj Patidar**
