import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score





MODEL_PATH = r"D:\My thesis\codes\iot_ids_dl_realtime\models\iot_ids_model.pth"
CSV_PATH = r"D:\My thesis\codes\iot_ids_dl_realtime\data\part0000.csv"
THRESHOLD = 0.4


df = pd.read_csv(CSV_PATH)


df["label"] = df["label"].apply(lambda x: 0 if x == "BenignTraffic" else 1)

X = df.drop("label", axis=1).values
y = df["label"].values


scaler = joblib.load(r"D:\My thesis\codes\iot_ids_dl_realtime\models\scaler.save")
X_scaled = scaler.transform(X)



model = nn.Sequential(
    nn.Linear(X_scaled.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)


model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Model loaded ✅")
print("Starting real-time simulation... ✅")


y_true = []
y_pred = []

N = 2000  

with torch.no_grad():
    for i in range(N):

        x_row = torch.tensor(X_scaled[i], dtype=torch.float32).view(1, -1)

        logit = model(x_row)
        prob = torch.sigmoid(logit).item()

        pred = 1 if prob >= THRESHOLD else 0
        true = int(y[i])

        y_true.append(true)
        y_pred.append(pred)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{N} rows")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


print(f"\nPrecision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

accuracy = sum([1 for a, b in zip(y_true, y_pred) if a == b]) / N
print(f"\nReal-time simulation accuracy: {accuracy*100:.2f}%")
print("Done ✅")

results = f"""
Confusion Matrix:
{confusion_matrix(y_true, y_pred)}

Precision: {precision*100:.2f}%
Recall: {recall*100:.2f}%
F1-score: {f1*100:.2f}%
Accuracy: {accuracy*100:.2f}%
"""

with open(r"D:\My thesis\codes\iot_ids_dl_realtime\results\final_results.txt", "w") as f:
    f.write(results)

print("Results saved successfully ✅")


