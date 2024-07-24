import os
import csv
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

def save_plot_height(df, original, path, y_lim=None):
    plt.figure(figsize=(12, 6))
    plt.plot(df['situation'], df['h_true'], label='h True')
    
    if original:
        plt.plot(df['situation'], df['h_estimation'], label='H Estimation 1')
    else:
        plt.plot(df['situation'], df['corrected_h_estimation'], label='h Corrected')

    plt.xlabel('Situation')
    plt.ylabel('Depth')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    if y_lim != None:
        plt.ylim(y_lim[0], y_lim[1])  # Establecer los límites del eje y

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
    plt.close() 


def extract_h_true(situation):
    return int(situation.split("_")[0])

laser =  pd.read_csv(f"data/alturas_train2.csv")
laser = laser.sort_values(["h_true"]).reset_index(drop=True)
laser['correction_factor'] = laser['h_true'] / laser['h_estimation']
def linearReg(df):

    X = df[['z_estimation']]
    y = df['correction_factor']

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    return model

model = linearReg(laser)
# Predicciones del modelo
laser['predicted_correction_factor'] = model.predict(laser[['z_estimation']])

# Aplicar el factor de corrección predicho a `h_estimation`
laser['corrected_h_estimation'] = laser['h_estimation'] * laser['predicted_correction_factor']

# Imprimir resultados
# print(laser[['situation', 'h_estimation', 'h_true', 'z_estimation', 'corrected_h_estimation']])

df_alturas_val = pd.read_csv(f"data/h_val.csv")
df_alturas_val = df_alturas_val.sort_values(["situation"]).reset_index(drop=True)
df_alturas_val["h_true"] = 173
# df_alturas_val["h_true"] = df_alturas_val["situation"].apply(extract_h_true)
df_alturas_val['predicted_correction_factor'] = model.predict(df_alturas_val[['z_estimation']])
df_alturas_val['corrected_h_estimation'] = df_alturas_val['h_estimation'] * df_alturas_val['predicted_correction_factor']
df_alturas_val["error"] = (abs(df_alturas_val["corrected_h_estimation"] - df_alturas_val["h_true"])*100)/df_alturas_val["h_true"]
print(df_alturas_val[['situation', 'h_estimation', 'h_true', 'z_estimation', 'corrected_h_estimation', 'error']])
print(np.mean(df_alturas_val['error'][:7]))

laser.to_excel(f'data/matlab_1/tables/heights/laser2.xlsx', index=False)
save_plot_height(laser, True, 'graficas/matlab_1/heights/laser2_train.png')
df_alturas_val.to_excel(f'data/matlab_1/tables/heights/laser2_c.xlsx', index=False)
save_plot_height(df_alturas_val, True, 'graficas/matlab_1/heights/laser2.png')
save_plot_height(df_alturas_val, False, 'graficas/matlab_1/heights/laser2_c.png', y_lim=(0,200))

# # Gráfica del factor de corrección en función de la profundidad
# plt.scatter(df_alturas_val['z_estimation'], df_alturas_val['correction_factor'], color='blue', label='Actual')
# plt.plot(df_alturas_val['z_estimation'], df_alturas_val['predicted_correction_factor'], color='red', linewidth=2, label='Predicted')
# plt.xlabel('z_estimation')
# plt.ylabel('Correction Factor')
# plt.legend()
# plt.show()

# # Gráfica de h_estimation corregido vs h_true
# plt.scatter(df_alturas_val['h_true'], df_alturas_val['corrected_h_estimation'], color='green', label='Corrected h_estimation')
# plt.plot([df_alturas_val['h_true'].min(), df_alturas_val['h_true'].max()], [df_alturas_val['h_true'].min(), df_alturas_val['h_true'].max()], color='black', linewidth=2, label='Ideal')
# plt.xlabel('h_true')
# plt.ylabel('Corrected h_estimation')
# plt.legend()
plt.show()

