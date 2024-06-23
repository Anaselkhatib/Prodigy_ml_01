# PRODIGY_ML_01: House Price Prediction Using Linear Regression

This project implements a linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)

## Overview
The goal of this project is to build a predictive model that estimates the prices of houses based on specific features. We use linear regression, a popular and straightforward machine learning algorithm, to achieve this.

## Dataset
The dataset used in this project includes the following features for each house:
- Square footage (`GrLivArea`)
- Number of bedrooms (`BedroomAbvGr`)
- Number of bathrooms (total from `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`)
- Price (`SalePrice`, target variable)

The dataset is preprocessed to handle missing values, normalize the data, and split into training and testing sets.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for visualization)

You can install the required packages using:
```bash
pip install pandas numpy scikit-learn matplotlib

Installation

    Clone the repository:

    bash

git clone https://github.com/Anaselkhatib/PRODIGY_ML_01.git
cd PRODIGY_ML_01

Install the required packages:

bash

    pip install -r requirements.txt

Usage

    Ensure your dataset is in the correct format (CSV recommended) and place it in the project directory.
    Run the preprocessing script to prepare the data:

    bash

python preprocess_data.py

Train the linear regression model:

bash

python train_model.py

Evaluate the model's performance:

bash

    python evaluate_model.py

Model Training

The model is trained using the LinearRegression class from the scikit-learn library. The training process includes:

    Loading and preprocessing the dataset.
    Splitting the data into training and testing sets.
    Training the model on the training set.
    Evaluating the model on the testing set using metrics such as Mean Squared Error (MSE) and R-squared.

Example code snippet for training the model:

python

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming X_train and Y_train are prepared
model = LinearRegression()
model.fit(X_train, Y_train)

y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(Y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

print(f"Training MSE: {train_mse:.2f}, Training RMSE: {train_rmse:.2f}")

Results

The performance of the model is evaluated based on:

    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE)
    R-squared (coefficient of determination)

Example output:

yaml

Training MSE: 1460715662.52, Training RMSE: 38219.31

Visualization of the results can be generated using matplotlib.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

arduino


Feel free to customize this template further based on your project details.

