# src/evaluate_model.py

import joblib
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import load_dataset, preprocess_data
from config import TEST_DATA_PATH, MODEL_PATH

def evaluate():
    model = joblib.load(MODEL_PATH)

    test_data = load_dataset(TEST_DATA_PATH)
    X_test, y_test = preprocess_data(test_data)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    evaluate()