# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -------------------------------------------------
# Load NSL-KDD Dataset (Fixed Version)
# -------------------------------------------------
def load_dataset(path):

    column_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files",
        "num_outbound_cmds","is_host_login","is_guest_login","count",
        "srv_count","serror_rate","srv_serror_rate","rerror_rate",
        "srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"
    ]

    # Read dataset safely
    data = pd.read_csv(
        path,
        names=column_names,
        header=None
    )

    # Remove empty rows
    data.dropna(inplace=True)

    print("Dataset loaded successfully ✅")
    print("Dataset shape:", data.shape)

    # Drop difficulty column
    if "difficulty" in data.columns:
        data.drop("difficulty", axis=1, inplace=True)

    return data


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_data(data):

    # Convert label to binary (Normal = 0, Attack = 1)
    data["label"] = data["label"].apply(
        lambda x: 0 if str(x).strip() == "normal" else 1
    )

    # Encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Separate features and label
    X = data.drop("label", axis=1)
    y = data["label"]

    # Check if dataset empty
    if X.shape[0] == 0:
        raise ValueError("Dataset is empty after preprocessing!")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y