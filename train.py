"""Train the model and save everything the app needs into model.joblib."""
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from preprocess import CAT_COLS, NUM_COLS, clean, encode

raw = pd.read_csv("loan_approval_data.csv")
df = clean(raw)

# --- metadata the UI needs (so dropdowns/sliders come from YOUR data, not hardcoded) ---
meta = {
    "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS},
    "num_stats": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                      "median": float(df[c].median())} for c in NUM_COLS + ["Credit_Score"]},
    "credit_score_median": float(df["Credit_Score"].median()),
}

df = encode(df)
X, y = df.drop("Loan_Approved", axis=1), df["Loan_Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

# --- same model as your updated notebook (ccp_alpha is the value you found) ---
CCP_ALPHA = 0.001316678691197293
model = DecisionTreeClassifier(max_depth=7, min_samples_split=5,
                               ccp_alpha=CCP_ALPHA, random_state=44)
model.fit(X_train, y_train)

# --- evaluate on the test set ---
pred = model.predict(X_test)
meta["metrics"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
print(meta["metrics"])

meta["feature_cols"] = X.columns.tolist()
meta["importances"] = dict(zip(X.columns, model.feature_importances_.tolist()))

joblib.dump({"model": model, "meta": meta}, "model.joblib")
print("saved model.joblib")
