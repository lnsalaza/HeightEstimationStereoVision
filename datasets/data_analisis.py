import csv
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# file_name = "z_estimation_opencv_1_keypoint5"
folder = "matlab_1" 
file_name = f"z_estimation_{folder}_keypoint" 
# file_name = "z_estimation_matlab_kp_cm" 

df = pd.read_csv(f"data/{folder}/{file_name}.csv")
df_gt = pd.read_csv(f"data/{folder}/ground_truth/{file_name}.csv")

# df = pd.read_csv(f"steven/{file_name}.csv")
# df = pd.read_csv("z_estimation_old_keypoints_no_astype_no_norm.csv")

df = pd.DataFrame(df)

def extract_z_true(situation):
    return int(situation.split("_")[0])

def extract_situation_2(situation):
    situation2 = int(situation.split("_"))
    if situation:
        return situation2

def apply_linear_correction(z_true):
    z_corrected = 0.4461*z_true + 3.445 
    return z_corrected*2

# LINEAR REGRESION PROCESS
def lineaRegression(true_values, estimated_values):
    model = LinearRegression()
    model.fit(np.array(estimated_values).reshape(-1,1), np.array(true_values).reshape(-1,1))
    
 
    predicted_values = model.predict(np.array(estimated_values).reshape(-1,1))
    return model, predicted_values

def linearCorrection(model, value):
    corrected_value = model.predict(np.array([[value]]))
    return corrected_value[0][0]

def apply_linear_regresion(df, col_x, col_y):
    X = df[[col_x]].values.reshape(-1,1)
    Y = df[col_y].values

    modelo = LinearRegression()
    modelo.fit(X,Y)
    modelo.predict(X)
    filename = f'models/{file_name}_ln_model.pkl'
    joblib.dump(modelo, filename)
    return modelo

# Función para verificar si una cadena contiene dos números
def has_two_numbers(s):
    parts = s.split('_')
    numbers = [part for part in parts if part.isdigit()]
    return len(numbers) == 2

def process_dataframe(df):
    # Identificar todas las columnas de z_estimation
    z_columns = [col for col in df.columns if col.startswith('z_estimation_')]
    
    # Si solo hay z_estimation_1, retornar el dataframe original
    if len(z_columns) == 1:
        return df
    
    # Filtrar las filas que cumplen con los requisitos
    df_filtered = df[df['situation'].apply(has_two_numbers)]
    
    new_rows = []
    
    for _, row in df_filtered.iterrows():
        parts = row['situation'].split('_')
        number1 = parts[0]
        number2 = parts[1]
        rest = '_' + '_'.join(parts[2:]) if len(parts) > 2 else ''
        
        for i in range(1, len(z_columns)):
            col_current = z_columns[i-1]
            col_next = z_columns[i]
            if pd.notna(row[col_next]):
                # new_rows.append({'situation': f"{number1}_{number2}{rest}", 'z_estimation_1': row[col_current], 'z_estimation_2': None})
                new_rows.append({'situation': f"{number2}_{number1}{rest}", 'z_estimation_1': row[col_next], 'z_estimation_2': None})
    
    df_new_rows = pd.DataFrame(new_rows)
    
    # Concatenar las nuevas filas al dataframe original (sin las filas que ya fueron procesadas)
    df_result = pd.concat([df, df_new_rows], ignore_index=True)
    return df_result


df["z_true"] = df["situation"].apply(extract_z_true)



# # TRAINING
# df_front = df[df["situation"].str.contains("front")]

# # VALIDATION
# df_variant = df[df["situation"].str.contains("variant")]

# # df_variant["z_corrected"] = df_variant["z_true"].apply(apply_linear_correction)

# lr_model = apply_linear_regresion(df_front, "z_estimation_1", "z_true")

## VARIANT DATAFRAME
#df_variant["z_corrected"] =  lr_model.predict(df_variant[["z_estimation_1"]].values.reshape(-1,1))
#df_variant["error"] = df_variant["z_true"] - df_variant["z_corrected"]
#print(df_variant)
## COMPLETE DATAFRAME
# df["z_corrected"] =  lr_model.predict(df[["z_estimation_1"]].values.reshape(-1,1))
# df["error"] = df["z_true"] - df["z_corrected"]
# print(df)

df_processed = process_dataframe(df)
df_processed["z_true"] = df_processed["situation"].apply(extract_z_true)
df_processed = df_processed.sort_values(["z_true"])

df_gt_processed = process_dataframe(df_gt)
df_gt_processed["z_true"] = df_gt_processed["situation"].apply(extract_z_true)
df_gt_processed = df_gt_processed.sort_values(["z_true"])


# TRAINING
df_front = df_processed[df_processed["situation"].str.contains("front")]


# df_variant["z_corrected"] = df_variant["z_true"].apply(apply_linear_correction)

#lr_model = apply_linear_regresion(df_processed, "z_estimation_1", "z_true")
lr_model = joblib.load("models/z_estimation_matlab_1_keypoint_ln_model_LASER.pkl")

df_gt_processed["z_corrected"] =  lr_model.predict(df_gt_processed[["z_estimation_1"]].values.reshape(-1,1))
df_gt_processed["error"] = (abs(df_gt_processed["z_corrected"] - df_gt_processed["z_true"])*100)/df_gt_processed["z_true"]

print(df_processed)
print('-----------------------------------------------------------------------------')
print(df_gt_processed)
print(np.mean(df_gt_processed['error'][:7]))
# VALIDATION
df_variant = df_processed[df_processed["situation"].str.contains("variant")]

# # GRAFICS

def save_plot(df, original, path):
    plt.figure(figsize=(12, 6))
    plt.plot(df['situation'], df['z_true'], label='z True',)
    if (original):
        plt.plot(df['situation'], df['z_estimation_1'], label = 'Z Estimation 1')
    else:
        plt.plot(df['situation'], df['z_corrected'], label = 'z Corrected')
 

    plt.xlabel('Situation')
    plt.ylabel('Depth')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    # plt.savefig(f"./steven/graficas/original_{file_name}.png")

# ###########################################ORIGINAL#####################################

# save_plot(df_front, True, f"./graficas/{folder}/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# save_plot(df_variant, False, f"./graficas/{folder}/corrected_{file_name}.png")

# ########################################### ORIGINAL COMPLETE #####################################

# save_plot(df_processed, True, f"./graficas/{folder}/original_{file_name}_complete.png")

# ########################################### CORRECTED COMPLETE #####################################

# save_plot(df_processed, False, f"./graficas/{folder}/corrected_{file_name}_complete.png")

###########################################ORIGINAL#####################################

save_plot(df_processed, True, f"./graficas/{folder}/ground_truth/original_{file_name}.png")

###########################################CORRECTED#####################################

save_plot(df_gt_processed, False, f"./graficas/{folder}/ground_truth/corrected_{file_name}.png")

