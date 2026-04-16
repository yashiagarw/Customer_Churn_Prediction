import joblib
import pandas as pd
from src.train_model import prepare_data, build_pipeline # might not exist
import os

base = os.path.dirname(os.path.abspath('app.py'))
models_dir = os.path.join(base, "models")
dt_model = joblib.load(os.path.join(models_dir, "churn_dt_model.joblib"))
scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.joblib"))

df = pd.DataFrame([{col: 0 for col in feature_cols}])
scaled = scaler.transform(df)
print(dt_model.predict_proba(scaled))
