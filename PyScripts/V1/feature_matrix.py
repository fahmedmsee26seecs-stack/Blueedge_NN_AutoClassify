import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"F:/NUST_MS_EE_ICS/Unsynced_Stuff/Sem1/AIML/BlueEdge_Project/Datasets/improved_dataset v4.csv")
df["sample"] = df["sample"].fillna("").astype(str)

X = np.array([extract(t) for t in df["sample"]], dtype=np.float32)
le = LabelEncoder()
y = le.fit_transform(df["data_type"])

print(X.shape, y.shape)      # expect (1400, 40) (1400,)
print(list(le.classes_))