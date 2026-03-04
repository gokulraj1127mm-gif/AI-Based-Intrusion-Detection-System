# src/predict.py

import joblib
import numpy as np
from config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict_sample(sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return "Attack" if prediction[0] == 1 else "Normal"