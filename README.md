# Elastic Net Regression - Concrete Strength Prediction

## Overview

This project implements an **Elastic Net Regression Model** to predict the compressive strength of concrete based on its composition and age.

The model is trained using a custom dataset and deployed through a **Flask** web application, allowing users to input concrete mix details and get instant predictions.

## Project Structure

```
DataScience/
│
├── Elastic Net/
│   ├── data/
│   │   └── concrete_dataset.csv
│   ├── model/
│   │   └── elastic_net_model.joblib
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── elastic_net_model.py
│   ├── app.py
│   └── requirements.txt
```

## Installation & Setup

1.  **Clone the repository**

    ```
    git clone <your-repo-url>
    cd "DataScience/Elastic Net"
    ```

2.  **Create a virtual environment (recommended)**

    ```
    python -m venv venv
    source venv/bin/activate    # For Linux/Mac
    venv\Scripts\activate      # For Windows
    ```

3.  **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

## Dataset

The dataset contains various components of concrete and their corresponding compressive strength.

* **Cement** (numeric): Amount of cement in the mix (kg/m³)
* **BlastFurnaceSlag** (numeric): Amount of blast furnace slag (kg/m³)
* **FlyAsh** (numeric): Amount of fly ash (kg/m³)
* **Water** (numeric): Amount of water (kg/m³)
* **Superplasticizer** (numeric): Amount of superplasticizer (kg/m³)
* **CoarseAggregate** (numeric): Amount of coarse aggregate (kg/m³)
* **FineAggregate** (numeric): Amount of fine aggregate (kg/m³)
* **Age** (numeric): Age of the concrete (days)
* **Strength** (Target): The compressive strength of the concrete (MPa)

## Problem Statement

Predicting the compressive strength of concrete is vital for construction. This project automates the process by providing a reliable way to estimate a concrete mix's strength based on its composition, which can save time and resources.

## Why Elastic Net Regression?

* **Combines Lasso and Ridge:** Elastic Net effectively combines the L1 (Lasso) and L2 (Ridge) regularization penalties. This gives it the best of both worlds.
* **Handles Multicollinearity:** The L2 penalty helps to stabilize the model when features are highly correlated, a common issue in this type of dataset.
* **Performs Feature Selection:** The L1 penalty can shrink the coefficients of less important features to zero, simplifying the model and improving its interpretability.
* **Robustness:** This combined approach makes the model more robust and less prone to overfitting, especially with a large number of features.

## How to Run

1.  **Train the Model**

    ```
    python elastic_net_model.py
    ```

    This will create:

    * `elastic_net_model.joblib` (trained model and scaler)

2.  **Run the Flask App**

    ```
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

## Frontend Input Example

Example concrete mix input:

```
Cement (kg/m³): 540.0
Blast Furnace Slag (kg/m³): 0.0
Fly Ash (kg/m³): 0.0
Water (kg/m³): 162.0
Superplasticizer (kg/m³): 2.5
Coarse Aggregate (kg/m³): 1040.0
Fine Aggregate (kg/m³): 676.0
Age (days): 28
```

## Prediction Goal

The application predicts the compressive strength of the concrete in Megapascals (MPa).

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **HTML/CSS** – Frontend UI design

## Future Scope

* Deploy the model on a cloud platform like Heroku or Render for public access.
* Implement a hyperparameter tuning script to find the optimal `alpha` and `l1_ratio` values for the Elastic Net model.
* Add a visualization to the result page that shows the predicted strength compared to a benchmark.


## Screen Shots

**Home Page:**

<img width="1920" height="1080" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/bb7d6c96-4810-46db-8907-2c97f72624aa" />


**Result Page:**

<img width="1920" height="1080" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/2a42c900-7f3e-458e-a19f-1f5ac3a1e1d6" />
