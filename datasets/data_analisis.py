import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression



df = pd.read_csv("z_estimation_old_keypoints.csv")

print(df)

def extract_z_true(situation):
    return int(situation.split("_")[0])

def extract_situation_2(situation):
    situation2 = int(situation.split("_"))
    if situation:
        return situation2
df["z_true"] = df["situation"].apply(extract_z_true)

df = df[df["situation"].str.contains("front")]



plt.figure(figsize=(12, 6))
plt.plot(df['situation'], df['z_true'], label='z True',)
plt.plot(df['situation'], df['z_estimation_1'], label = 'Z Estimation 1')

plt.xlabel('Situation')
plt.ylabel('Depth')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig("now2.png")



# LINEAR REGRESION PROCESS


print(df)