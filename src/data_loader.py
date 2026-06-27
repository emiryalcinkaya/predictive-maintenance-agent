import pandas as pd


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)