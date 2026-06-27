from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from analysis import analyze_dataset
from data_loader import load_data
from evaluate_model import evaluate_model
from preprocessing import preprocess_data


def main():

    # Load dataset
    df = load_data("data/ai4i2020.csv")

    # Analyze dataset
    analyze_dataset(df)

    # Prepare dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    print("\n=== Decision Tree Model ===")
    print("Decision Tree model trained successfully.")

    evaluate_model(
        decision_tree,
        X_test,
        y_test
    )

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    print("\n=== Random Forest Model ===")
    print("Random Forest model trained successfully.")

    evaluate_model(
        random_forest,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()