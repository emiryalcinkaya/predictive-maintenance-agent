import pandas as pd

from llm_advisor import get_maintenance_advice
from model_utils import load_model


def main():

    print("=== Predictive Maintenance Agent ===\n")

    # Load trained model
    model = load_model("random_forest.joblib")

    # Get machine information from the user
    machine_type = input("Machine Type (L/M/H): ").upper()

    air_temperature = float(input("Air Temperature [K]: "))
    process_temperature = float(input("Process Temperature [K]: "))
    rotational_speed = int(input("Rotational Speed [rpm]: "))
    torque = float(input("Torque [Nm]: "))
    tool_wear = int(input("Tool Wear [min]: "))

    # Prepare input data for prediction
    input_data = pd.DataFrame([
        {
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type_H": 1 if machine_type == "H" else 0,
            "Type_L": 1 if machine_type == "L" else 0,
            "Type_M": 1 if machine_type == "M" else 0,
        }
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Calculate prediction confidence
    confidence = max(probability) * 100

    print("\n===== Prediction =====")

    if prediction == 1:
        prediction_text = "Machine Failure"
    else:
        prediction_text = "No Machine Failure"

    print(prediction_text)
    print(f"\nConfidence: {confidence:.2f}%")

    print("\n===== AI Maintenance Recommendation =====")

    # Generate maintenance advice using Llama 3
    advice = get_maintenance_advice(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        prediction_text,
        confidence
    )

    print(advice)


if __name__ == "__main__":
    main()