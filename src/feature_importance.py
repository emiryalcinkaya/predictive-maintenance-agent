import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):
    """
    Plot and save feature importance.
    """

    # Calculate feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    # Sort features
    importance = importance.sort_values()

    # Create images folder if it does not exist
    os.makedirs("images", exist_ok=True)

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot feature importance
    importance.plot(kind="barh")

    plt.title(f"{type(model).__name__} Feature Importance")
    plt.xlabel("Importance")

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "images/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Feature Importance saved successfully.")