import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# file_name = "z_estimation_opencv_1_keypoint" 
file_name = "z_estimation_matlab_kp_cm" 

df = pd.read_csv(f"steven/{file_name}.csv")
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
    return modelo


df["z_true"] = df["situation"].apply(extract_z_true)

# TRAINING
df_front = df[df["situation"].str.contains("front")]

# VALIDATION
df_variant = df[df["situation"].str.contains("variant")]

# df_variant["z_corrected"] = df_variant["z_true"].apply(apply_linear_correction)

lr_model = apply_linear_regresion(df_front, "z_estimation_1", "z_true")

#df_variant["z_corrected"] =  lr_model.predict(df_variant[["z_estimation_1"]].values.reshape(-1,1))


#df_variant["error"] = df_variant["z_true"] - df_variant["z_corrected"]

df["z_corrected"] =  lr_model.predict(df[["z_estimation_1"]].values.reshape(-1,1))


df["error"] = df["z_true"] - df["z_corrected"]


print(df)
# # GRAFICS

# ###########################################ORIGINAL#####################################

# plt.figure(figsize=(12, 6))
# plt.plot(df_front['situation'], df_front['z_true'], label='z True',)
# plt.plot(df_front['situation'], df_front['z_estimation_1'], label = 'Z Estimation 1')


# plt.xlabel('Situation')
# plt.ylabel('Depth')
# plt.xticks(rotation=90)
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# # plt.savefig(f"./graficas/original_{file_name}.png")
# plt.savefig(f"./steven/graficas/original_{file_name}.png")

# ###########################################CORRECTED#####################################

# plt.figure(figsize=(12, 6))
# plt.plot(df_variant['situation'], df_variant['z_true'], label='z True',)
# plt.plot(df_variant['situation'], df_variant['z_corrected'], label = 'z Corrected')


# plt.xlabel('Situation')
# plt.ylabel('Depth')
# plt.xticks(rotation=90)
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# # plt.savefig(f"./graficas/corrected_{file_name}.png")
# plt.savefig(f"./steven/graficas/corrected_{file_name}.png")


###########################################CORRECTED#####################################

plt.figure(figsize=(12, 6))
plt.plot(df['situation'], df['z_true'], label='z True',)
plt.plot(df['situation'], df['z_corrected'], label = 'z Corrected')


plt.xlabel('Situation')
plt.ylabel('Depth')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig(f"./graficas/corrected_{file_name}.png")
plt.savefig(f"./steven/graficas/corrected_{file_name}_prueba.png")
