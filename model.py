import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import joblib

# --- Constants ---
# Assuming the dataset is in a 'data' directory relative to this script.
DATA_PATH = 'dataset/concrete_dataset.csv'
MODEL_PATH = 'model/elastic_net_model.joblib'

# --- 1. Load the Dataset ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{DATA_PATH}' was not found. Please ensure the dataset is in the correct path.")
    exit()

# --- 2. Separate Features (X) and Target (y) ---
# The last column 'Strength' is the target variable.
# All other columns are features.
X = df.drop('Strength', axis=1)
y = df['Strength']

# --- 3. Split Data into Training and Testing Sets ---
# We use a 80/20 split for training and testing.
# A random state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 4. Scale the Features ---
# Regularization methods like Elastic Net are sensitive to feature scales.
# Scaling ensures all features contribute equally to the penalty term.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

# --- 5. Instantiate and Train the ElasticNet Model ---
# Elastic Net combines L1 and L2 regularization.
# `alpha`: Controls the overall strength of the regularization.
# `l1_ratio`: Controls the balance between L1 (Lasso) and L2 (Ridge) penalties.
# `l1_ratio = 1` is Lasso, `l1_ratio = 0` is Ridge. We will use a value in between.
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

print("Training the Elastic Net model...")
elastic_net_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
# Predict on the test data to see how well the model performs on unseen data.
y_pred = elastic_net_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"Model R^2 score on the test set: {r2:.4f}")

# --- 7. Save the Trained Model and Scaler ---
# It's good practice to save the trained model and the scaler.
# The scaler is needed to preprocess any new data before making predictions.
model_pipeline = {
    'model': elastic_net_model,
    'scaler': scaler
}
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model and scaler saved to '{MODEL_PATH}'.")

print("\nModel training and saving process finished.")
print("You can now use this saved model for making new predictions.")
