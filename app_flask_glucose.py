from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ======= ML + Signal Processing Code =========

def extract_ppg_signal(video_path):
    if not os.path.exists(video_path):
        return np.array([])

    cap = cv2.VideoCapture(video_path)
    ppg_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[0] < 300 or frame.shape[1] < 300:
            continue

        roi = frame[100:300, 100:300]
        red_channel = roi[:, :, 2]
        brightness = np.mean(red_channel)
        ppg_signal.append(brightness)

    cap.release()
    return np.array(ppg_signal)

def extract_features(ppg_signal):
    if len(ppg_signal) == 0:
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

# Train a model once on startup
X_train, y_train = generate_mock_data()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============ API Route ============

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    file_path = os.path.join("uploads", file.filename)

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Process and predict
    signal = extract_ppg_signal(file_path)
    features = extract_features(signal)

    if features.empty:
        return jsonify({"error": "Failed to extract features"}), 400

    prediction = model.predict(features)[0]
    return jsonify({"glucose_level": float(prediction)})

# ============ Run App ============

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
