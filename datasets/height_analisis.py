import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

CONFIG = "matlab_1"
METHOD_USED = "RAFT" #OPTIONS: "SGBM". "RAFT", "SELECTIVE"


FILE_NAME = f"z_estimation_heights_ground_truth_keypoint"
FILE_NAME_VAL = f"corrected_z_estimation_validation_keypoint"

def save_plot_height(df, original, path, y_lim=None):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['situation'], df['h_true'], label='h True')
        
        label = 'H Estimation 1' if original else 'h Corrected'
        estimation_col = 'h_estimation' if original else 'corrected_h_estimation'
        plt.plot(df['situation'], df[estimation_col], label=label)

        plt.xlabel('Situation')
        plt.ylabel('Depth')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        
        if y_lim:
            plt.ylim(*y_lim)  # Desempaquetar los límites del eje y

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    except FileNotFoundError:
        print(f"Error: The directory specified in the path {path} does not exist.")
    except KeyError as e:
        print(f"Error: Missing column in DataFrame: {e}")
    except Exception as e:
        print(f"Unexpected error in save_plot_height: {e}")

def extract_x_true(situation):
    try:
        return int(situation.split("_")[0])*10 if METHOD_USED != 'SGBM' else int(situation.split("_")[0])
    except IndexError:
        raise ValueError(f"Invalid situation format, expected at least one '_': {situation}")
    except ValueError:
        raise ValueError(f"Invalid situation format, expected integer conversion: {situation}")
    except Exception as e:
        print(f"Unexpected error in extract_x_true: {e}")

def linear_regression(x, y):
    try:
        model = LinearRegression()
        model.fit(x, y)
        return model
    except ValueError as e:
        print(f"Error in linear_regression: {e}")
    except Exception as e:
        print(f"Unexpected error in linear_regression: {e}")

def save_model(model, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(model, filename)
    except FileNotFoundError:
        print(f"Error: The directory specified in the filename {filename} does not exist.")
    except Exception as e:
        print(f"Unexpected error in save_model: {e}")

def load_data(filepath, sort_value):
    try:
        return pd.read_csv(filepath).sort_values([sort_value]).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
    except KeyError:
        print(f"Error: The sort value {sort_value} is not a valid column in the DataFrame.")
    except Exception as e:
        print(f"Unexpected error in load_data: {e}")


def preprocess_training_data(df):
    try:
        df['correction_factor'] = df['h_true'] / df['h_estimation']
        df['z_true'] = df['situation'].apply(extract_x_true)
        return df
    except KeyError as e:
        print(f"Error: Missing column in DataFrame during preprocessing: {e}")
    except Exception as e:
        print(f"Unexpected error in preprocess_training_data: {e}")

def calculate_error(df):
    try:
        df["error"] = (abs(df["corrected_h_estimation"] - df["h_true"]) * 100) / df["h_true"]
        return df, np.mean(df['error'])
    except KeyError as e:
        print(f"Error: Missing column in DataFrame during error calculation: {e}")
    except Exception as e:
        print(f"Unexpected error in calculate_error: {e}")
# Load and preprocess training data
training_data_filepath = f"data/{CONFIG}/{METHOD_USED}/{FILE_NAME}_train.csv"
training_data = load_data(training_data_filepath, "h_true")
training_data = preprocess_training_data(training_data)

# Train model
# model = linear_regression(training_data[['z_true']], training_data['correction_factor'])
# save_model(model, './models/matlab_1/height/LASER2/h_gt_alturas_model.pkl')

# Load and preprocess validation data
# validation_data_filepath = f"data/{CONFIG}/{METHOD_USED}/{FILE_NAME_VAL}_validation.csv"
# validation_data = load_data(validation_data_filepath, "situation")
# validation_data["h_true"] = 1730.0 if METHOD_USED != 'SGBM' else 173.0
# validation_data = validation_data[validation_data['z_estimation'] >= 176.0]

# z_estimation = validation_data[['z_estimation']].rename(columns={'z_estimation': 'z_true'})
# validation_data['predicted_correction_factor'] = model.predict(z_estimation)
# validation_data['corrected_h_estimation'] = validation_data['h_estimation'] * validation_data['predicted_correction_factor']
# validation_data, error = calculate_error(validation_data)

save_plot_height(training_data, True, f"graficas/{CONFIG}/heights/{METHOD_USED}/training_{FILE_NAME}.png")
# save_plot_height(validation_data, True, f"graficas/{CONFIG}/heights/{METHOD_USED}/original_{FILE_NAME}_val.png")
# save_plot_height(validation_data, False, f"graficas/{CONFIG}/heights/{METHOD_USED}/corrected_{FILE_NAME}_val.png")

print(training_data)
# print(validation_data)
# print(error)
# Save data and plotss
# training_data.to_excel('data/matlab_1/tables/heights/laser2.xlsx', index=False)
# save_plot_height(training_data, True, 'graficas/matlab_1/heights/laser2_train.png')
# validation_data.to_excel('data/matlab_1/tables/heights/laser2_c.xlsx', index=False)
# save_plot_height(validation_data, True, 'graficas/matlab_1/heights/laser2.png')
# save_plot_height(validation_data, False, 'graficas/matlab_1/heights/laser2_c.png', y_lim=(0, 200))
