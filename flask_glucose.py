import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

print("All libraries imported successfully!")

# 1. Extract PPG-like signal from fingertip video
def extract_ppg_signal(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video file '{video_path}' not found.")
        return np.array([])

    cap = cv2.VideoCapture(video_path)
    ppg_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Safety check for frame shape
        h, w, _ = frame.shape
        if h < 300 or w < 300:
            print("❌ Frame too small for ROI. Skipping.")
            continue

        roi = frame[100:300, 100:300]  # Region of interest
        red_channel = roi[:, :, 2]  # Red channel
        brightness = np.mean(red_channel)
        ppg_signal.append(brightness)

    cap.release()

    if len(ppg_signal) == 0:
        print("❌ No signal extracted.")
    return np.array(ppg_signal)

# 2. Feature Extraction
def extract_features(ppg_signal):
    if len(ppg_signal) == 0:
        print("❌ Empty PPG signal! Cannot extract features.")
        return pd.DataFrame()

    features = {
        "mean": np.mean(ppg_signal),
        "std": np.std(ppg_signal),
        "min": np.min(ppg_signal),
        "max": np.max(ppg_signal),
        "range": np.max(ppg_signal) - np.min(ppg_signal),
        "slope": np.mean(np.diff(ppg_signal)),
    }
    return pd.DataFrame([features])

# 3. Generate mock data
def generate_mock_data(n=100):
    X = pd.DataFrame({
        "mean": np.random.normal(150, 5, n),
        "std": np.random.normal(3, 0.5, n),
        "min": np.random.normal(120, 5, n),
        "max": np.random.normal(180, 5, n),
        "range": np.random.normal(60, 3, n),
        "slope": np.random.normal(0.01, 0.005, n),
    })
    y = 70 + 0.2 * X["mean"] + 0.3 * X["range"] + np.random.normal(0, 5, n)
    return X, y

# 4. Train model
def train_model():
    X, y = generate_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained. Test MSE:", mean_squared_error(y_test, model.predict(X_test)))
    return model

# 5. Predict glucose level
def predict_glucose(video_path, model):
    signal = extract_ppg_signal(video_path)
    features = extract_features(signal)

    if features.empty:
        print("❌ Feature extraction failed. Cannot predict.")
        return

    prediction = model.predict(features)[0]
    print(f"🩸 Predicted Blood Glucose Level: {prediction:.2f} mg/dL")
    return prediction

# 6. Main function
if __name__ == "__main__":
    model = train_model()
    video_file = "finger_vid3.mp4"  # Replace with your own video file
    predict_glucose(video_file, model)