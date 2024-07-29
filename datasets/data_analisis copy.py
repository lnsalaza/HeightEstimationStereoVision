import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import utils

# Constantes
FOLDER = "matlab_1"
FILE_NAME = f"z_estimation_{FOLDER}_keypoint"

# Funciones de utilidades
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")

def extract_z_true(situation):
    try:
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

def save_plot(df, col_x, col_y, path, title):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df[col_x], df[col_y], marker='o', linestyle='-', color='b', label=col_y)
        plt.xlabel(col_x.capitalize())
        plt.ylabel(col_y.capitalize())
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.title(title)
        plt.tight_layout()
        utils.create_directory(path)
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")


# Carga de datos
df_training = load_data(f"data/{FOLDER}/ground_truth/{FILE_NAME}_training.csv")
df_validation = load_data(f"data/{FOLDER}/{FILE_NAME}_validation.csv")

# Preprocesamiento
df_processed = process_dataframe(df_training, 'z_estimation_')
df_processed["z_true"] = df_processed["situation"].apply(extract_z_true)
df_processed = df_processed.sort_values("z_true")

df_validation["z_true"] = df_validation["situation"].apply(extract_z_true)
df_validation = df_validation.sort_values("z_true")

# Entrenamiento y validación
lr_model = apply_linear_regression(df_processed, "z_estimation_1", "z_true")
utils.save_model(lr_model, f'models/{FOLDER}/{FILE_NAME}.pkl')

df_validation["z_corrected"] = lr_model.predict(df_validation[["z_estimation_1"]].values.reshape(-1, 1))
df_validation, validation_error = utils.calculate_error(df_validation, "z_true", "z_corrected")

# Guardar resultados
df_validation.to_excel(f'data/{FOLDER}/tables/LASER2.xlsx', index=False)
print(df_validation)
print(f"Validation error: {validation_error:.2f}%")

# Guardar gráficos
# save_plot(df_processed, "situation", "z_true", f"graficas/{FOLDER}/original_{FILE_NAME}.png", "Original Z True")
# save_plot(df_validation, "situation", "z_corrected", f"graficas/{FOLDER}/corrected_{FILE_NAME}.png", "Corrected Z Estimation")

# ###########################################ORIGINAL#####################################

# save_plot(df_front, True, f"./graficas/{folder}/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# save_plot(df_variant, False, f"./graficas/{folder}/corrected_{file_name}.png")

# ########################################### ORIGINAL COMPLETE #####################################

# save_plot(df_processed, True, f"./graficas/{folder}/original_{file_name}_complete.png")

# ########################################### CORRECTED COMPLETE #####################################

# save_plot(df_processed, False, f"./graficas/{folder}/corrected_{file_name}_complete.png")

###########################################ORIGINAL#####################################

# save_plot(df_processed, True, f"./graficas/{folder}/ground_truth/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# graficar_alturas(df_alturas_train, 0, 250)