import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display evaluation results
    print("\n===== Model Evaluation =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Display classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Create images folder if it does not exist
    os.makedirs("images", exist_ok=True)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"]
    )

    # Plot confusion matrix
    disp.plot(cmap="Blues")

    plt.title(f"{type(model).__name__} Confusion Matrix")

    # Save confusion matrix
    plt.savefig(
        f"images/{type(model).__name__.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion Matrix saved successfully.")