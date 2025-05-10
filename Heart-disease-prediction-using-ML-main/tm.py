import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset (Replace with actual dataset)
df = pd.read_csv("heart.csv")  # Ensure the dataset file is present

# Features and target
X = df.drop("target", axis=1)  # Drop the target column
y = df["target"]  # Target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model as model.pkl
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl successfully!")
