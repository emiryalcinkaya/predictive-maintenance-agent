import os

import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(df):
    """
    Display basic information and perform exploratory data analysis (EDA).
    """

    # Display dataset size
    print("=== Dataset Shape ===")
    print(df.shape)

    # Display column names
    print("\n=== Columns ===")
    print(df.columns.tolist())

    # Display first rows
    print("\n=== First 5 Rows ===")
    print(df.head())

    # Display dataset information
    print("\n=== Dataset Information ===")
    df.info()

    # Display statistical summary
    print("\n=== Statistical Summary ===")
    print(df.describe())

    # Display class distribution
    print("\n=== Machine Failure Distribution ===")
    print(df["Machine failure"].value_counts())

    # Display class percentages
    print("\n=== Machine Failure Percentage (%) ===")
    print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    # Display missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    # Create images folder if it does not exist
    os.makedirs("images", exist_ok=True)

    # EDA 1 - Class Distribution
    plt.figure(figsize=(6, 4))

    ax = sns.countplot(
        data=df,
        x="Machine failure"
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Count")

    plt.xticks(
        [0, 1],
        ["No Failure", "Failure"]
    )

    plt.tight_layout()

    plt.savefig(
        "images/class_distribution.png",
        dpi=300
    )

    plt.close()

    print("\nClass Distribution plot saved successfully.")

    # EDA 2 - Correlation Heatmap
    correlation_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df[correlation_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "images/correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    print("Feature Correlation Heatmap saved successfully.")