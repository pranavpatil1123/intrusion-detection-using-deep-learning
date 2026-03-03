import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"D:\My thesis\codes\iot_ids_dl_realtime\data\part0000.csv")

df["label"] = df["label"].apply(lambda x: 0 if x == "BenignTraffic" else 1)

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    
)


pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)


loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 3
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training finished ✅")

X_test_t = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    
    logits = model(X_test_t)
    preds = torch.sigmoid(logits)
    preds_class = (preds >= 0.3).float()


y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

accuracy = (preds_class.eq(y_test_t).sum().item()) / y_test_t.shape[0]

print("Test Accuracy:", accuracy)


final_threshold = 0.4

preds_class = (preds >= final_threshold).float()

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

y_true = y_test_t.numpy().astype(int).ravel()
y_pred = preds_class.numpy().astype(int).ravel()

print("\nFinal Threshold:", final_threshold)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))

import os

save_path = r"D:\My thesis\codes\iot_ids_dl_realtime\models\iot_ids_model.pth"

torch.save(model.state_dict(), save_path)

print("Model saved successfully ✅")

import joblib

joblib.dump(scaler, r"D:\My thesis\codes\iot_ids_dl_realtime\models\scaler.save")

print("Scaler saved ✅")
