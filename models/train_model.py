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
from pathlib import Path
import pandas as pd

# Define the paths
main_file = Path("../data/credit_risk_dataset.csv")
sample_file = Path("data/sample_data.csv")

# Load the appropriate file
if main_file.exists():
    df = pd.read_csv(main_file)
    print(f"Loaded main dataset: {main_file}")
elif sample_file.exists():
    df = pd.read_csv(sample_file)
    print(f"Main dataset not found. Loaded sample dataset: {sample_file}")
else:
    raise FileNotFoundError("Neither main nor sample dataset found.")

# Ensure target column exists
if "target" not in df.columns:
    raise ValueError("The loaded dataset must contain a 'target' column")


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
