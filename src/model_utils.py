import os
import joblib


def save_model(model, filename):
    """
    Save the trained model to the models folder.
    """

    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Create file path
    filepath = os.path.join("models", filename)

    # Save model
    joblib.dump(model, filepath)

    print(f"\nModel saved successfully: {filepath}")


def load_model(filename):
    """
    Load a trained model from the models folder.
    """

    # Create file path
    filepath = os.path.join("models", filename)

    # Load model
    model = joblib.load(filepath)

    print(f"\nModel loaded successfully: {filepath}")

    return model