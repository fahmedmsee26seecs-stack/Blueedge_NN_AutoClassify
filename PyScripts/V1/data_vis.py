import pandas as pd
df = pd.read_csv("F:/NUST_MS_EE_ICS/Unsynced_Stuff/Sem1/AIML/BlueEdge_Project/Datasets/improved_dataset v4.csv")
print(df.shape)
print(df["data_type"].value_counts())
print(df.groupby("data_type").head(2))