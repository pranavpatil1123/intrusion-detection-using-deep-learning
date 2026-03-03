import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\My thesis\codes\iot_ids_dl_realtime\data\part0000.csv")

df["label"] = df["label"].apply(lambda x: 0 if x == "BenignTraffic" else 1)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)       

print("Scaling Done ✅")
print("Training shape:", X_train.shape)
