# Gluco Vision

## 📦 Features

- Extracts PPG signals from video using color channels and region of interest.
- Applies **Butterworth bandpass** and **Savitzky-Golay smoothing** for noise reduction.
- Computes physiological features like **heart rate**, **entropy**, and **signal area**.
- Generates **synthetic glucose data** (normal + diabetic distributions).
- Combines real and synthetic data to train a **Gradient Boosting Regressor**.
- Outputs glucose prediction with simple risk level warnings.


##Install Dependencies 
```bash
pip install opencv-python numpy pandas scipy scikit-learn

##📈 Example Output
✅ Model trained with enhanced synthetic and real data. Test MSE: 92.7
🩸 Estimated Glucose: 142.3 mg/dL
🔺 Elevated glucose - Monitor or consult physician

##🧪 Future Improvements
Add face detection and tracking

Support multiple video formats

Integrate with real-world datasets

Improve model performance with neural networks

