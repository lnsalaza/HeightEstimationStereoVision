import os
import joblib
import numpy as np


def create_directory(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def save_model(model, filename):
    try:
        create_directory(filename)
        joblib.dump(model, filename)
    except Exception as e:
        print(f"Error saving model: {e}")

def calculate_error(df, col_true, col_pred):
    try:
        df["error"] = ((abs(df[col_pred] - df[col_true])) / df[col_true])
        return df, np.mean(df["error"])
    except KeyError as e:
        print(f"Error: Missing column in DataFrame: {e}")

