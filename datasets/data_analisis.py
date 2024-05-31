import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt 

df = pd.read_csv("z_estimation_old_keypoints.csv")
def extract_z_true(situation):
    return int(situation.split("_")[0])
df["z_true"] = df["situation"].apply(extract_z_true)

df = df[df["situation"].str.contains("front")]


print(df)

plt.figure(figsize=(12, 6))
plt.plot(df['situation'], df['z_true'], label='z True',)
plt.plot(df['situation'], df['z_estimation_1'], label = 'Z Estimation 1')

plt.xlabel('Situation')
plt.ylabel('Depth')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("old_config2.png")