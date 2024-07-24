import os
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

# df = pd.read_csv(f"data/{folder}/ground_truth/{file_name}_training.csv") 
# df_gt = pd.read_csv(f"data/{file_name}_validation.csv")
df = pd.read_csv(f"data/{folder}/z_estimation_train.csv") 


df_gt = pd.read_csv(f"data/{folder}/{file_name}_validation.csv")

# df_alturas_train = pd.read_csv(f"data/{file_name}_h_train.csv")
# df_alturas_train = df_alturas_train.sort_values(["situation"]).reset_index(drop=True)

# df_alturas_val = pd.read_csv(f"data/{file_name}_heights_val.csv")
# df_alturas_val = df_alturas_val.sort_values(["situation"]).reset_index(drop=True)

# df_alturas_corr = pd.read_csv(f"data/{file_name}_heights_corrected.csv")
# df_alturas_corr = df_alturas_corr.sort_values(["situation"]).reset_index(drop=True)

# df = pd.read_csv(f"steven/{file_name}.csv")
# df = pd.read_csv("z_estimation_old_keypoints_no_astype_no_norm.csv")

def separador(section=''):
    print(f'----------------------------------{section}-------------------------------------------')

def graficar_alturas(df, altura_minima, altura_maxima):
    """
    Función para graficar alturas estimadas.

    :param alturas_estimadas: Lista de alturas estimadas.
    :param altura_minima: Valor mínimo del rango de altura.
    :param altura_maxima: Valor máximo del rango de altura.
    """
    alturas_estimadas = df["h_estimation_1"]
    # Creación de la figura y los ejes
    fig, ax = plt.subplots()

    # Gráfico de la línea de alturas estimadas
    ax.plot(alturas_estimadas, marker='o', linestyle='-', color='b')

    # Configuración de los límites de los ejes
    
    ax.set_xlim(0, len(alturas_estimadas) - 1)
    ax.set_ylim(altura_minima, altura_maxima)

    # Etiquetas de los ejes
    ax.set_xlabel('Situacion')
    ax.set_ylabel('Altura estimada (cm)')

    ax.set_xticks(range(len(df["situation"])))
    ax.set_xticklabels(df["situation"], rotation=90, ha="right")

    # Título de la gráfica
    ax.set_title('Comportamiento de las alturas estimadas')
    plt.tight_layout()
    plt.savefig("IMG_alturas")
    plt.close()

def extract_z_true(situation):
    return int(situation.split("_")[0])

def extract_h_true(situation):
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

def apply_linear_regresion(df, col_x, col_y, filename):
    X = df[[col_x]].values.reshape(-1,1)
    Y = df[col_y].values

    modelo = LinearRegression()
    modelo.fit(X,Y)
    modelo.predict(X)
    filename = f'models/{filename}.pkl'

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    joblib.dump(modelo, filename)
    return modelo

# Función para verificar si una cadena contiene dos números
def has_two_numbers(s):
    parts = s.split('_')
    numbers = [part for part in parts if part.isdigit()]
    return len(numbers) == 2

def process_dataframe(df, column):
    # Identificar todas las columnas de z_estimation
    z_columns = [col for col in df.columns if col.startswith(column)]
    
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
                new_rows.append({'situation': f"{number2}_{number1}{rest}", f'{column}1': row[col_next]})
    
    df_new_rows = pd.DataFrame(new_rows)
    
    # Concatenar las nuevas filas al dataframe original (sin las filas que ya fueron procesadas)
    df_result = pd.concat([df, df_new_rows], ignore_index=True)
    return df_result

# TRAINING
# df_processed = process_dataframe(df, 'z_estimation_')
df_processed = df
df_processed["z_true"] = df_processed["situation"].apply(extract_z_true)
df_processed = df_processed.sort_values(["z_true"])

# VALIDATION
# df_gt_processed = process_dataframe(df_gt, 'z_estimation_')
df_gt_processed = df_gt
df_gt_processed["z_true"] = df_gt_processed["situation"].apply(extract_z_true)
df_gt_processed = df_gt_processed.sort_values(["z_true"])


# lr_model = apply_linear_regresion(df_processed, "z_estimation", "z_true", f'{folder}/z_estimation_lr')
lr_model = joblib.load("models/matlab_1/z_estimation_lr.pkl")

# df_gt_processed["z_corrected"] =  lr_model.predict(df_gt_processed[["z_estimation"]].values.reshape(-1,1))
df_gt_processed["z_corrected"] =  lr_model.predict(df_gt_processed[["z_estimation_1"]])
df_gt_processed["error"] = (abs(df_gt_processed["z_corrected"] - df_gt_processed["z_true"]))/df_gt_processed["z_true"]

# separador()
df_gt_processed.to_excel(f'data/{folder}/tables/z_estimation_corrected.xlsx', index=False)
print(df_gt_processed)
print(np.mean(df_gt_processed['error'])*100)



# print('-----------------------------------------------------------------------------')
# df_alturas_train["h_true"] = df_alturas_train["situation"].apply(extract_h_true)
# df_alturas_val["h_true"] = 173
# h_model = apply_linear_regresion(df_alturas_train, "h_estimation_1", "h_true", f'{folder}/height_lr')
# df_alturas_val["h_corrected"] = h_model.predict(df_alturas_val[["h_estimation_1"]].values.reshape(-1,1))
# df_alturas_val["error"] = (abs(df_alturas_val["h_corrected"] - df_alturas_val["h_true"])*100)/df_alturas_val["h_true"]
# file = f'data/{folder}/tables/heights_lr/'
# if not os.path.exists(os.path.dirname(file)):
#     os.makedirs(os.path.dirname(file))
# separador('ALTURAS TRAIN')
# # print(df_alturas_train)
# df_alturas_train.to_excel(f'{file}/{file_name}_train.xlsx', index=False)
# separador('ALTURAS VAL')
# # print(df_alturas_val)
# df_alturas_val.to_excel(f'{file}/{file_name}_val.xlsx', index=False)

# df_variant = df_processed[df_processed["situation"].str.contains("variant")]

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
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
    # plt.savefig(f"./steven/graficas/original_{file_name}.png")

def save_plot_height(df, original, path):
    plt.figure(figsize=(12, 6))
    plt.plot(df['situation'], df['h_true'], label='h True',)
    if (original):
        plt.plot(df['situation'], df['h_estimation'], label = 'H Estimation 1')
    else:
        plt.plot(df['situation'], df['h_corrected'], label = 'h Corrected')
 

    plt.xlabel('Situation')
    plt.ylabel('Depth')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
    # plt.savefig(f"./steven/graficas/original_{file_name}.png")



# cinta = pd.read_csv(f"data/cinta.csv")
# cinta_c =  pd.read_csv(f"data/cinta_corregido.csv")
# laser =  pd.read_csv(f"data/laser.csv")
# laser_c =  pd.read_csv(f"data/laser_corregido.csv")

# cinta["error"] = (abs(cinta["h_estimation"] - cinta["h_true"])*100)/cinta["h_true"]
# cinta = cinta.sort_values(["h_true"]).reset_index(drop=True)
# cinta_c["error"] = (abs(cinta_c["h_estimation"] - cinta_c["h_true"])*100)/cinta_c["h_true"]
# cinta_c = cinta_c.sort_values(["h_true"]).reset_index(drop=True)
# laser["error"] = (abs(laser["h_estimation"] - laser["h_true"])*100)/laser["h_true"]
# laser = laser.sort_values(["h_true"]).reset_index(drop=True)
# laser_c["error"] = (abs(laser_c["h_estimation"] - laser_c["h_true"])*100)/laser_c["h_true"]
# laser_c = laser_c.sort_values(["h_true"]).reset_index(drop=True)

# file = f'data/{folder}/tables/heights/'
# if not os.path.exists(os.path.dirname(file)):
#     os.makedirs(os.path.dirname(file))
# cinta.to_excel(f'{file}/cinta.xlsx', index=False)
# cinta_c.to_excel(f'{file}/cinta_c.xlsx', index=False)
# laser.to_excel(f'{file}/laser.xlsx', index=False)
# laser_c.to_excel(f'{file}/laser_c.xlsx', index=False)

# file = f'graficas/{folder}/heights/'
# if not os.path.exists(os.path.dirname(file)):
#     os.makedirs(os.path.dirname(file))
# save_plot_height(cinta, True, f'{file}/cinta.png')
# save_plot_height(cinta_c, True, f'{file}/cinta_c.png')
# save_plot_height(laser, True, f'{file}/laser.png')
# save_plot_height(laser_c, True, f'{file}/laser_c.png')

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

# save_plot(df_gt_processed, False, f"./graficas/{folder}/ground_truth/corrected_{file_name}.png")
# print('depth')
# print(df_processed)
# print('-----------------------------------------------------------------------------')
# print(df_gt_processed)
# print('-----------------------------------------------------------------------------')
# print('height')
# print(df_alturas_train)
# print('-----------------------------------------------------------------------------')
# print(df_alturas_val)

# df_alturas_corr['h_true'] = 173
# df_alturas_corr["error"] = (abs(df_alturas_corr["h_corrected"] - df_alturas_corr["h_true"])*100)/df_alturas_corr["h_true"]

# print(df_alturas_corr.shape)

# save_plot_height(df_alturas_train, True, f"./graficas/{folder}/heights/original_train_{file_name}.png")
# save_plot_height(df_alturas_val, True, f"./graficas/{folder}/heights/original_val_{file_name}.png")
# save_plot_height(df_alturas_val, False, f"./graficas/{folder}/heights/corrected_train_{file_name}.png")
# save_plot_height(df_alturas_corr, False, f"./graficas/{folder}/heights/corrected_val_{file_name}.png")

# graficar_alturas(df_alturas_train, 0, 250)