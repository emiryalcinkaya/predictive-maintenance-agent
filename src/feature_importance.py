import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values()

    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 5))

    importance.plot(kind="barh")

    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Feature Importance saved successfully.")