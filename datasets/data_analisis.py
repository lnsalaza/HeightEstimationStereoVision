import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import utils

# Constantes
CONFIG = "matlab_1"
METHOD_USED = "SELECTIVE" #OPTIONS: "SGBM". "RAFT", "SELECTIVE"


FILE_NAME = f"z_estimation_chessboard_ground_truth_keypoint"
FILE_NAME_VAL = f"z_estimation_validation_keypoint"

# CONFIG = "matlab_1/ground_truth"
# FILE_NAME = f"z_estimation_matlab_1_keypoint"
# FILE_NAME_VAL = f"SGBM_z_estimation_validation_keypoint"

# Funciones de utilidades
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")

def extract_z_true(situation):
    try:
        if METHOD_USED != 'SGBM':
            return int(situation.split("_")[0])*10
        return int(situation.split("_")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid situation format: {situation}")

def linea_regression(true_values, estimated_values):
    try:
        model = LinearRegression()
        model.fit(np.array(estimated_values).reshape(-1,1), np.array(true_values).reshape(-1,1))
        predicted_values = model.predict(np.array(estimated_values).reshape(-1,1))
        return model, predicted_values
    except Exception as e:
        print(f"Error in linear regression: {e}")

def apply_linear_regression(df, col_x, col_y):
    try:
        X = df[[col_x]].values.reshape(-1, 1)
        Y = df[col_y].values
        model = LinearRegression()
        model.fit(X, Y)
        return model
    except Exception as e:
        print(f"Error applying linear regression: {e}")


def has_two_numbers(s):
    parts = s.split('_')
    numbers = [part for part in parts if part.isdigit()]
    return len(numbers) == 2

def process_dataframe(df, column_prefix):
    z_columns = [col for col in df.columns if col.startswith(column_prefix)]
    if len(z_columns) == 1:
        return df
    
    df_filtered = df[df['situation'].apply(has_two_numbers)]
    new_rows = []

    for _, row in df_filtered.iterrows():
        parts = row['situation'].split('_')
        number1, number2 = parts[:2]
        rest = '_' + '_'.join(parts[2:]) if len(parts) > 2 else ''

        for i in range(1, len(z_columns)):
            col_current = z_columns[i-1]
            col_next = z_columns[i]
            if pd.notna(row[col_next]):
                new_rows.append({'situation': f"{number2}_{number1}{rest}", f'{column_prefix}1': row[col_next]})

    df_new_rows = pd.DataFrame(new_rows)
    return pd.concat([df, df_new_rows], ignore_index=True)

def save_plot(df, original, path, y_lim=None):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['situation'], df['z_true'], label='z True')
        
        label = 'Z Estimation' if original else 'Z Corrected'
        estimation_col = 'z_estimation' if original else 'z_corrected'
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


# Carga de datos
df_training = load_data(f"data/{CONFIG}/{METHOD_USED}/{FILE_NAME}_train.csv")
df_validation = load_data(f"data/{CONFIG}/{METHOD_USED}/{FILE_NAME_VAL}_validation.csv")

# Preprocesamiento
df_processed = process_dataframe(df_training, 'z_estimation')
df_processed["z_true"] = df_processed["situation"].apply(extract_z_true)
df_processed = df_processed.sort_values("z_true")

df_validation["z_true"] = df_validation["situation"].apply(extract_z_true)
df_validation = df_validation.sort_values("z_true")

# Entrenamiento y validación
# lr_model = joblib.load('models/matlab_1/depth/z_LASER2.pkl')
lr_model = apply_linear_regression(df_processed, "z_estimation", "z_true")
utils.save_model(lr_model, f'models/{CONFIG}/depth/{METHOD_USED}.pkl')

df_validation["z_corrected"] = lr_model.predict(df_validation[["z_estimation"]].values.reshape(-1, 1))
df_validation, validation_error = utils.calculate_error(df_validation, "z_true", "z_corrected")

# Guardar resultados
df_validation.to_excel(f'data/matlab_1/tables/{FILE_NAME_VAL}.xlsx', index=False)
print(df_validation)
print(f"Validation error: {validation_error:.2f}%")

# Guardar gráficos
save_plot(df_processed, True, f"graficas/{CONFIG}/{METHOD_USED}/training_{FILE_NAME}.png")
save_plot(df_validation, True, f"graficas/{CONFIG}/{METHOD_USED}/original_{FILE_NAME}_val.png")
save_plot(df_validation, False, f"graficas/{CONFIG}/{METHOD_USED}/corrected_{FILE_NAME}_val.png")

# ###########################################ORIGINAL#####################################

# save_plot(df_front, True, f"./graficas/{CONFIG}/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# save_plot(df_variant, False, f"./graficas/{CONFIG}/corrected_{file_name}.png")

# ########################################### ORIGINAL COMPLETE #####################################

# save_plot(df_processed, True, f"./graficas/{CONFIG}/original_{file_name}_complete.png")

# ########################################### CORRECTED COMPLETE #####################################

# save_plot(df_processed, False, f"./graficas/{CONFIG}/corrected_{file_name}_complete.png")

###########################################ORIGINAL#####################################

# save_plot(df_processed, True, f"./graficas/{CONFIG}/ground_truth/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# graficar_alturas(df_alturas_train, 0, 250)