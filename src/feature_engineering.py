# src/feature_engineering.py

from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd


# -----------------------------------------
# Remove Constant Features
# -----------------------------------------
def remove_constant_features(data):
    """
    Remove columns that have only one unique value
    """
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    data = data.drop(columns=constant_cols)
    return data


# -----------------------------------------
# Select Top K Features (FIXED VERSION)
# -----------------------------------------
def select_top_features(X, y, k=25):
    """
    Select top k important features using ANOVA F-test.
    Works with negative values (unlike chi2).
    """

    # Ensure k is not larger than available features
    if k > X.shape[1]:
        k = X.shape[1]

    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    print(f"Selected Top {k} Features ✅")

    return X_new, selector