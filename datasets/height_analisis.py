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

def extract_x_true(situation):
    return int(situation.split("_")[0])

def linearReg(x,y, filename):

    X = x
    y = y

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    joblib.dump(model, filename)
    return model

### Modelo GT y Alturas ***EL BUENO***
"""
El mejor modelo fue entrenado con datos de ground truth de laser y dotos de alturas, ademas se utilizó
entreno con el z_true.
"""
laser1 =  pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/h_train_gt_alturas.csv")
laser1 = laser1.sort_values(["h_true"]).reset_index(drop=True)
laser1['correction_factor'] = laser1['h_true'] / laser1['h_estimation']

laser1['z_true'] = laser1['situation'].apply(extract_x_true)
model1 = linearReg(laser1[['z_true']],laser1['correction_factor'], './models/matlab_1/height/LASER2/h_gt_alturas_model.pkl' )

df_val1 = pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/validation-z_corrected-LASER2_model-.csv")
df_val1 = df_val1.sort_values(["situation"]).reset_index(drop=True)
df_val1["h_true"] = 173
df_val1 = df_val1.drop(df_val1[df_val1['z_estimation']<176.0].index)
z_estimation = df_val1[['z_estimation']].rename(columns={'z_estimation':'z_true'})
df_val1['predicted_correction_factor'] = model1.predict(z_estimation)
df_val1['corrected_h_estimation'] = df_val1['h_estimation'] * df_val1['predicted_correction_factor']
df_val1["error"] = (abs(df_val1["corrected_h_estimation"] - df_val1["h_true"])*100)/df_val1["h_true"]
error1 = np.mean(df_val1['error'])

# laser.to_excel(f'data/matlab_1/tables/heights/laser2.xlsx', index=False)
# save_plot_height(laser, True, 'graficas/matlab_1/heights/laser2_train.png')
# df_val1.to_excel(f'data/matlab_1/tables/heights/laser2_c.xlsx', index=False)
# save_plot_height(df_val1, True, 'graficas/matlab_1/heights/laser2.png')
# save_plot_height(df_val1, False, 'graficas/matlab_1/heights/laser2_c.png', y_lim=(0,200))

plt.show()

