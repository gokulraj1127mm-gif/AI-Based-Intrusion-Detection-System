# src/train_model.py

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_preprocessing import load_dataset, preprocess_data
from src.feature_engineering import remove_constant_features
from src.utils import save_model, log_message

from config import TRAIN_DATA_PATH


def train():

    log_message("Loading dataset...")
    data = load_dataset(TRAIN_DATA_PATH)

    log_message("Removing constant features...")
    data = remove_constant_features(data)

    log_message("Preprocessing data...")
    X, y = preprocess_data(data)

    log_message("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_message("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    log_message("Evaluating model...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print("\nTraining Completed ✅")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    save_model(model, "models/trained_model.pkl")

    print("Model saved successfully 🚀")


if __name__ == "__main__":
    train()