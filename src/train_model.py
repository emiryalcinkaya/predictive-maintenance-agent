from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from analysis import analyze_dataset
from data_loader import load_data
from evaluate_model import evaluate_model
from feature_importance import plot_feature_importance
from model_utils import save_model
from preprocessing import preprocess_data


def main():
    """
    Train and evaluate the machine learning models.
    """

    # Load dataset
    df = load_data("data/ai4i2020.csv")

    # Analyze dataset
    analyze_dataset(df)

    # Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Decision Tree model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Save Decision Tree model
    save_model(
        decision_tree,
        "decision_tree.joblib"
    )

    print("\n=== Decision Tree Model ===")
    print("Decision Tree model trained successfully.")

    # Generate feature importance plot
    print("\nGenerating Feature Importance...")

    plot_feature_importance(
        decision_tree,
        X_train.columns
    )

    # Evaluate Decision Tree model
    evaluate_model(
        decision_tree,
        X_test,
        y_test
    )

    # Train Random Forest model
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    # Save Random Forest model
    save_model(
        random_forest,
        "random_forest.joblib"
    )

    print("\n=== Random Forest Model ===")
    print("Random Forest model trained successfully.")

    # Evaluate Random Forest model
    evaluate_model(
        random_forest,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()