import pandas as pd


def load_frequency_data(path: str):
    return pd.read_csv(path)


def load_severity_data(path: str):
    return pd.read_csv(path)
