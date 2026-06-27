def analyze_dataset(df):
    """
    Display basic information and statistics about the dataset.
    """

    print("=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Dataset Information ===")
    df.info()

    print("\n=== Statistical Summary ===")
    print(df.describe())

    print("\n=== Machine Failure Distribution ===")
    print(df["Machine failure"].value_counts())

    print("\n=== Machine Failure Percentage (%) ===")
    print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    print("\n=== Missing Values ===")
    print(df.isnull().sum())