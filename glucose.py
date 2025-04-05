import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt
import wfdb

# -------------------------------
# 1. Load ECG Signal from PhysioNet MIT-BIH
# -------------------------------
def load_ecg_from_physionet(record_name="100", sample_count=1000):
    """
    Loads ECG signal from PhysioNet MIT-BIH Arrhythmia Database using WFDB.
    Applies bandpass filtering to denoise the signal.
    """
    print(f"📥 Loading ECG record: {record_name}")
    record = wfdb.rdrecord(record_name, pn_dir="mitdb")
    ecg_signal = record.p_signal[:, 0]  # Use channel 1

    # Trim and filter
    ecg_signal = ecg_signal[:sample_count]
    fs = 360  # Sampling rate (Hz)
    b, a = butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered_signal = filtfilt(b, a, ecg_signal)

    # Plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_signal)
    plt.title(f"Filtered ECG Signal - Record {record_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return filtered_signal

# -------------------------------
# 2. Feature Extraction
# -------------------------------
def extract_features(signal):
    """
    Extracts simple statistical features from ECG signal.
    """
    if len(signal) == 0:
        print("❌ Empty signal!")
        return pd.DataFrame()

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "range": np.ptp(signal),
        "slope": np.mean(np.diff(signal)),
    }

    print("✅ Extracted Features:", features)
    return pd.DataFrame([features])

# -------------------------------
# 3. Generate Mock Data
# -------------------------------
def generate_mock_data(n=300):
    """
    Generates synthetic data for training the model.
    """
    X = pd.DataFrame({
        "mean": np.random.normal(0, 0.5, n),
        "std": np.random.normal(0.1, 0.05, n),
        "min": np.random.normal(-0.5, 0.1, n),
        "max": np.random.normal(0.5, 0.1, n),
        "range": np.random.normal(1.0, 0.2, n),
        "slope": np.random.normal(0.001, 0.0005, n),
    })
    y = 80 + 0.3 * X["range"] + 0.2 * X["std"] + np.random.normal(0, 5, n)
    return X, y

# -------------------------------
# 4. Train the Model
# -------------------------------
def train_model():
    """
    Trains a Random Forest Regressor on synthetic ECG feature data.
    """
    X, y = generate_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"✅ Model trained. Test MSE: {test_mse:.2f}")
    return model

# -------------------------------
# 5. Run Prediction
# -------------------------------
def predict_glucose_ecg(record_name, model):
    """
    Pipeline: Load ECG → Extract features → Predict simulated glucose.
    """
    ecg_signal = load_ecg_from_physionet(record_name)
    features = extract_features(ecg_signal)

    if features.empty:
        print("❌ Cannot predict - feature extraction failed.")
        return None

    prediction = model.predict(features)[0]
    print(f"🩸 Predicted (Simulated) Glucose Level: {prediction:.2f} mg/dL")
    return prediction

# -------------------------------
# 6. Run the Script
# -------------------------------
if __name__ == "__main__":
    model = train_model()
    predict_glucose_ecg("100", model)  # Replace with other record IDs like "101", "102" etc.
