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

laser1 =  pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/h_train_gt_alturas.csv")
laser1 = laser1.sort_values(["h_true"]).reset_index(drop=True)
laser1['correction_factor'] = laser1['h_true'] / laser1['h_estimation']

laser1['z_true'] = laser1['situation'].apply(extract_x_true)
model1 = linearReg(laser1[['z_true']],laser1['correction_factor'], './models/matlab_1/height/LASER2/h_gt_alturas_model.pkl' )

df_val1 = pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/validation-z_corrected-LASER2_model-.csv")
df_val1 = df_val1.sort_values(["situation"]).reset_index(drop=True)
df_val1["h_true"] = 173
# df_val1 = df_val1.rename(columns={'z_estimation':'z_true'})
df_val1 = df_val1.drop(df_val1[df_val1['z_estimation']<176.0].index)
# df_val1["h_true"] = df_val1["situation"].apply(extract_x_true)
z_estimation = df_val1[['z_estimation']].rename(columns={'z_estimation':'z_true'})
df_val1['predicted_correction_factor'] = model1.predict(z_estimation)
df_val1['corrected_h_estimation'] = df_val1['h_estimation'] * df_val1['predicted_correction_factor']
df_val1["error"] = (abs(df_val1["corrected_h_estimation"] - df_val1["h_true"])*100)/df_val1["h_true"]
error1 = np.mean(df_val1['error'])

### Modelo GT 

laser2 =  pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/h_train_gt.csv")
laser2 = laser2.sort_values(["h_true"]).reset_index(drop=True)
laser2['correction_factor'] = laser2['h_true'] / laser2['h_estimation']

laser2['z_true'] = laser2['situation'].apply(extract_x_true)
model2 = linearReg(laser2[['z_true']],laser2['correction_factor'], './models/matlab_1/height/h_gt.pkl' )

df_val2 = pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/validation-z_corrected-LASER2_model-.csv")
df_val2 = df_val2.sort_values(["situation"]).reset_index(drop=True)
df_val2["h_true"] = 173
# df_val2 = df_val2.rename(columns={'z_estimation':'z_true'})
df_val2 = df_val2.drop(df_val2[df_val2['z_estimation']<176.0].index)
# df_val2["h_true"] = df_val2["situation"].apply(extract_x_true)
z_estimation = df_val2[['z_estimation']].rename(columns={'z_estimation':'z_true'})
df_val2['predicted_correction_factor'] = model2.predict(z_estimation)
df_val2['corrected_h_estimation'] = df_val2['h_estimation'] * df_val2['predicted_correction_factor']
df_val2["error"] = (abs(df_val2["corrected_h_estimation"] - df_val2["h_true"])*100)/df_val2["h_true"]
error2 = np.mean(df_val2['error'])

df_val4 = pd.read_csv(f"data/matlab_1/heights/h_correction_LASER/h_train_alturas.csv")
df_val4 = df_val4.sort_values(["situation"]).reset_index(drop=True)
df_val4["h_true"] = 173
# df_val4 = df_val4.rename(columns={'z_estimation':'z_true'})
df_val4 = df_val4.drop(df_val4[df_val4['z_estimation']<176.0].index)
# df_val4["h_true"] = df_val4["situation"].apply(extract_x_true)
z_estimation = df_val4[['z_estimation']].rename(columns={'z_estimation':'z_true'})
df_val4['predicted_correction_factor'] = model2.predict(z_estimation)
df_val4['corrected_h_estimation'] = df_val4['h_estimation'] * df_val4['predicted_correction_factor']
df_val4["error"] = (abs(df_val4["corrected_h_estimation"] - df_val4["h_true"])*100)/df_val4["h_true"]
error4 = np.mean(df_val4['error'])

### Modelo Alturas

laser3 =  pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/h_train_alturas.csv")
laser3 = laser3.sort_values(["h_true"]).reset_index(drop=True)
laser3['correction_factor'] = laser3['h_true'] / laser3['h_estimation']

laser3['z_true'] = laser3['situation'].apply(extract_x_true)
model3 = linearReg(laser3[['z_true']],laser3['correction_factor'], './models/matlab_1/height/h_alturas.pkl' )

df_val3 = pd.read_csv(f"data/matlab_1/heights/h_correction_LASER2/validation-z_corrected-LASER2_model-.csv")
df_val3 = df_val3.sort_values(["situation"]).reset_index(drop=True)
df_val3["h_true"] = 173
# df_val3 = df_val3.rename(columns={'z_estimation':'z_true'})
df_val3 = df_val3.drop(df_val3[df_val3['z_estimation']<176.0].index)
# df_val3["h_true"] = df_val3["situation"].apply(extract_x_true)
z_estimation = df_val3[['z_estimation']].rename(columns={'z_estimation':'z_true'})
df_val3['predicted_correction_factor'] = model3.predict(z_estimation)
df_val3['corrected_h_estimation'] = df_val3['h_estimation'] * df_val3['predicted_correction_factor']
df_val3["error"] = (abs(df_val3["corrected_h_estimation"] - df_val3["h_true"])*100)/df_val3["h_true"]
error3 = np.mean(df_val3['error'])

# laser.to_excel(f'data/matlab_1/tables/heights/laser2.xlsx', index=False)
# save_plot_height(laser, True, 'graficas/matlab_1/heights/laser2_train.png')
# df_val1.to_excel(f'data/matlab_1/tables/heights/laser2_c.xlsx', index=False)
# save_plot_height(df_val1, True, 'graficas/matlab_1/heights/laser2.png')
# save_plot_height(df_val1, False, 'graficas/matlab_1/heights/laser2_c.png', y_lim=(0,200))

# # Gráfica del factor de corrección en función de la profundidad
# plt.scatter(df_val1['z_estimation'], df_val1['correction_factor'], color='blue', label='Actual')
# plt.plot(df_val1['z_estimation'], df_val1['predicted_correction_factor'], color='red', linewidth=2, label='Predicted')
# plt.xlabel('z_estimation')
# plt.ylabel('Correction Factor')
# plt.legend()
# plt.show()

# # Gráfica de h_estimation corregido vs h_true
# plt.scatter(df_val1['h_true'], df_val1['corrected_h_estimation'], color='green', label='Corrected h_estimation')
# plt.plot([df_val1['h_true'].min(), df_val1['h_true'].max()], [df_val1['h_true'].min(), df_val1['h_true'].max()], color='black', linewidth=2, label='Ideal')
# plt.xlabel('h_true')
# plt.ylabel('Corrected h_estimation')
# plt.legend()
plt.show()

