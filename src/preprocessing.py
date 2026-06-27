import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """
    Prepare the dataset for machine learning.
    """

    selected_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]

    model_data = df[selected_columns].copy()

    # One-Hot Encode machine type
    model_data = pd.get_dummies(
        model_data,
        columns=["Type"],
        dtype=int
    )

    # Features and target
    X = model_data.drop("Machine failure", axis=1)
    y = model_data["Machine failure"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test