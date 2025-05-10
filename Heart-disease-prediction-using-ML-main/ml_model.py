import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("heart.pkl", "rb"))

def predict_heart_disease(features):
    """
    Takes a list of input features and predicts heart disease risk.
    :param features: List of input features (must match training format)
    :return: Prediction result (High Risk / Low Risk)
    """
    try:
        # Ensure the input is in the correct shape for the model
        input_data = np.array(features).reshape(1, -1)  # Reshape for model
        prediction = model.predict(input_data)  # Predict result

        # Return prediction result based on the model's output
        return "Low Risk of Heart Disease" if prediction[0] == 0 else "High Risk of Heart Disease"
    
    except Exception as e:
        return f"Error in ML processing: {str(e)}"
