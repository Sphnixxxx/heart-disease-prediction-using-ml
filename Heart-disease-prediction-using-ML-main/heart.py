import pandas as pd
import joblib  # If using a saved model
import sys

# Load your trained model (modify this based on your model type)
model = joblib.load("heart_model.pkl")  # Ensure this model file exists

def predict_heart_disease(input_csv):
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    # Ensure feature columns match the model's input format
    features = df.values  # Modify this if needed
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Print the result (so Flask can capture it)
    print(int(prediction))

if __name__ == "__main__":
    predict_heart_disease(sys.argv[1])  # Take CSV file input
