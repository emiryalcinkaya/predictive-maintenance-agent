import os
import joblib


def save_model(model, filename):

    os.makedirs("models", exist_ok=True)

    filepath = os.path.join("models", filename)

    joblib.dump(model, filepath)

    print(f"\nModel saved successfully: {filepath}")