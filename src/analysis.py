def analyze_dataset(df):
    """
    Display basic information and statistics about the dataset.
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