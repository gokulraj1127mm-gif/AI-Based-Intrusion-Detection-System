from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("..", "models", "trained_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get 40 features in correct order
        features = [float(value) for value in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)

        result = "Attack Detected 🚨" if prediction[0] == 1 else "Normal Traffic ✅"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)