# models/train_model.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/credit_risk_dataset.csv" if Path("../data/credit_risk_dataset.csv").exists() else "data/sample_data.csv")
if "target" not in df.columns:
    raise ValueError("data/sample_data.csv must contain a 'target' column")

X = df.drop(columns=["target"])
y = df["target"].astype(int)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# pipelines
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols),
])

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("Training pipeline...")
pipeline.fit(X_train, y_train)
print("Training done.")

# Save model
out_dir = Path("../models") if Path("../models").exists() else Path("models")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model.pkl"
joblib.dump(pipeline, out_path)
print("Saved trained pipeline to", out_path)
