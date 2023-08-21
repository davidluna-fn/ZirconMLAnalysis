# ZirconMLAnalysis

This repository contains the code for analyzing and predicting the price of zircon gemstones using machine learning techniques. The main goal of this project is to build a predictive model that can accurately estimate the price of zircon gemstones based on various features.

## Problem Statement

Zircon gemstones come in various shapes, sizes, and qualities, which can greatly affect their market value. The challenge is to develop a machine learning model that can predict the price of zircon gemstones based on features such as carat, cut, color, clarity, and dimensions. The model aims to provide accurate price estimates to both buyers and sellers.

## Objectives

1. Exploratory Data Analysis (EDA): Analyze the dataset to understand the relationships between different features and the target variable (price).

2. Data Preprocessing: Process and prepare the data for training the predictive model. This involves handling missing values, encoding categorical features, and creating relevant new features.

3. Model Training: Train a machine learning model to predict the price of zircon gemstones. Hyperparameter tuning and feature selection are performed to improve model performance.

4. API Deployment: Expose the trained model through a FastAPI-based API. Users can upload a dataset (in CSV or Excel format) to receive price predictions for the zircon gemstones.

## Project Structure

The project is structured as follows:

- `data/`: Contains processed and raw data directories.
- `models/`: Stores trained models and predictions.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `reports/`: Generated analysis reports and figures.
- `requirements.txt`: Required Python packages for reproducing the analysis environment.
- `src/`: Source code for data processing, feature engineering, model training, and API endpoints.
  - `data/`: Scripts to download or generate data.
  - `features/`: Scripts to preprocess and engineer features.
  - `models/`: Scripts for model training and prediction.
  - `app/`: Scripts for creating API endpoints.

## Getting Started

1. Clone this repository: `git clone https://github.com/davidluna-fn/ZirconMLAnalysis.git`
2. Navigate to the repository: `cd ZirconMLAnalysis`
3. Install the required packages: `pip install -r requirements.txt` only necessary if you are going to run notebooks
4. To build and run the Docker image:
   - Build the image: `docker build -t zircon-api .`
   - Run the container: `docker run -d -p 8000:8000 zircon-api`

5. Access the API at `http://localhost:8000/zircoins/api/V1/predict/`


## Trying the API

FastAPI's built-in documentation makes it straightforward to explore and test the API. Enjoy using the API to obtain price predictions for your zircon gemstones!

You can easily test the API and make price predictions for zircon gemstones by following these steps:

1. Open your web browser and navigate to: [http://localhost:8000/docs](http://localhost:8000/docs)

![](/reports/figures_test_api/f1.png)

2. In the Swagger UI interface provided by FastAPI, you will see a list of available endpoints. Select the "predict_url" endpoint.



3. Click the "Try it out" button.

![](/reports/figures_test_api/f3.png)


4. Click the "Choose File" button to upload an Excel file containing the data you want to predict.


5. After selecting the file, click the "Execute" button.

![](/reports/figures_test_api/f4.png)

6. The Swagger UI will show the response, which contains the predicted prices for the zircon gemstones in the uploaded file.


![](/reports/figures_test_api/f6.png)


## Author

This project was developed by David Luna Naranjo. Feel free to contact me at davidluna.fn@gmail.com for any questions or feedback.


