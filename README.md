# IoT Intrusion Detection System using Deep Learning

## Project Overview
This project implements a Deep Learning-based Intrusion Detection System (IDS) for IoT network traffic.  
The model detects whether incoming traffic is normal or an attack.

## Project Structure
- load_data.py → Data loading and preprocessing
- train_model.py → Model training
- realtime_test.py → Real-time prediction
- models/iot_ids_model.pth → Trained model
- models/scaler.save → Saved scaler

## How to Run

1. Install required libraries:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Test in real-time:
   python realtime_test.py
