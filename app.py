from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the trained model file
MODEL_PATH = 'model/elastic_net_model.joblib'

# Load the trained model and scaler
# The model and scaler are loaded once when the application starts
try:
    with open(MODEL_PATH, 'rb') as f:
        pipeline = joblib.load(f)
        elastic_net_model = pipeline['model']
        scaler = pipeline['scaler']
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: The model file '{MODEL_PATH}' was not found.")
    print("Please run 'model.py' first to train and save the model.")
    # Exit the application if the model file is missing to prevent runtime errors
    exit()

@app.route('/')
def home():
    """Renders the home page with the input form for concrete features."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request. It processes the form data, scales the input,
    and returns a prediction using the trained Elastic Net model.
    """
    try:
        # Get the input values from the form.
        # The keys here must match the 'name' attributes in the HTML form.
        features = [
            float(request.form['Cement']),
            float(request.form['BlastFurnaceSlag']),
            float(request.form['FlyAsh']),
            float(request.form['Water']),
            float(request.form['Superplasticizer']),
            float(request.form['CoarseAggregate']),
            float(request.form['FineAggregate']),
            float(request.form['Age'])
        ]
        
        # Convert the list of features into a NumPy array with the correct shape
        input_data = np.array(features).reshape(1, -1)

        # Scale the input data using the pre-trained scaler
        scaled_input = scaler.transform(input_data)

        # Make the prediction using the loaded Elastic Net model
        prediction = elastic_net_model.predict(scaled_input)[0]

        # Format the prediction to a more readable string with two decimal places
        formatted_prediction = f"{prediction:.2f} MPa"

        # Pass the prediction to the result page template
        return render_template('result.html', prediction=formatted_prediction)

    except ValueError:
        # Handle cases where the input is not a valid number
        error_message = "Invalid input. Please ensure all fields contain numerical values."
        return render_template('result.html', prediction=error_message)
    except Exception as e:
        # Handle any other unexpected errors during prediction
        error_message = f"An error occurred: {str(e)}"
        return render_template('result.html', prediction=error_message)

if __name__ == '__main__':
    # Run the application. debug=True allows for automatic reloading on code changes.
    app.run(debug=True)
