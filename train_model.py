# train_model.py
"""
Train a house price per sqft model and save:
 - models/hp_model_v1.pkl
 - models/feature_names.txt
 - models/X_test_sample.csv

This script automatically selects the correct OneHotEncoder argument
depending on your scikit-learn version (uses sparse_output for newer versions).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"Python: {sys.version.splitlines()[0]}")
print("scikit-learn version detected:", sklearn.__version__)

# ---------- Adjust these hints if needed ----------
POSSIBLE_TARGETS = ['Avg_Price_per_sqft', 'Price_per_sqft', 'Price']
NUMERIC_CANDIDATES = ['Area_sqft', 'BHK', 'bathrooms', 'Balcony']
CATEGORICAL_CANDIDATES = ['Locality', 'Furnishing', 'City', 'Type']
DATE_CANDIDATES = ['Year_Built', 'Year']
# --------------------------------------------------

# Load dataset (Excel or CSV)
if os.path.exists('new_dataset.xlsx'):
    df = pd.read_excel('new_dataset.xlsx', engine='openpyxl')
elif os.path.exists('new_dataset.csv'):
    df = pd.read_csv('new_dataset.csv')
else:
    raise FileNotFoundError("Place new_dataset.xlsx or new_dataset.csv in this folder.")

print("Dataset loaded. Shape:", df.shape)

# Detect target column
target = None
for t in POSSIBLE_TARGETS:
    if t in df.columns:
        target = t
        break

# If only Price and Area found → create price per sqft
if target is None and 'Price' in df.columns and 'Area_sqft' in df.columns:
    df['Avg_Price_per_sqft'] = df['Price'] / df['Area_sqft']
    target = 'Avg_Price_per_sqft'

if target is None:
    raise ValueError("Couldn't find target column. Edit POSSIBLE_TARGETS at top of script.")

print("Target chosen:", target)

# Build derived feature "age" if Year present
current_year = 2025
for yc in DATE_CANDIDATES:
    if yc in df.columns:
        try:
            df['age'] = current_year - pd.to_numeric(df[yc], errors='coerce')
            print("Derived 'age' from", yc)
        except Exception:
            pass
        break

# Prepare features
num_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

features = num_cols + cat_cols
if len(features) == 0:
    raise ValueError("No candidate feature columns found from the lists. Check NUMERIC_CANDIDATES/CATEGORICAL_CANDIDATES.")

print("Numeric cols used:", num_cols)
print("Categorical cols used:", cat_cols)

# Keep only rows with non-null in features + target
data = df[features + [target]].copy().dropna()
if data.shape[0] == 0:
    raise ValueError("No rows left after dropping NA for chosen features and target. Inspect dataset.")

X = data[features]
y = data[target].replace([np.inf, -np.inf], np.nan).dropna()
X = X.loc[y.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Select correct OneHotEncoder argument based on sklearn version
try:
    skl_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
except Exception:
    skl_ver = (1, 4)  # assume modern

if skl_ver >= (1, 4):
    # sklearn 1.4+ expects sparse_output
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    print("Using OneHotEncoder(..., sparse_output=False)")
else:
    # older sklearn uses sparse
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    print("Using OneHotEncoder(..., sparse=False)")

# Preprocessing pipelines
num_pipeline = Pipeline([('scaler', StandardScaler())]) if num_cols else 'passthrough'
cat_pipeline = Pipeline([('ohe', ohe)]) if cat_cols else 'passthrough'

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
], remainder='drop')

# Model pipeline
model_pipeline = Pipeline([
    ('pre', preprocessor),
    ('reg', LGBMRegressor(n_estimators=300, random_state=42))
])

# Fit model
print("Fitting model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Results")
print("Test MAE :", round(mae, 2))
print("Test RMSE:", round(rmse, 2))
print("Test R²  :", round(r2, 3))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model_pipeline, 'models/hp_model_v1.pkl')

# Save one sample test CSV
try:
    X_test.sample(min(20, len(X_test))).to_csv('models/X_test_sample.csv', index=False)
except Exception:
    X_test.to_csv('models/X_test_sample.csv', index=False)

# Save feature names (best-effort)
feature_names = []
try:
    # Try sklearn's get_feature_names_out on ColumnTransformer
    feature_names = model_pipeline.named_steps['pre'].get_feature_names_out()
except Exception:
    # Fallback: numeric columns + expanded cat names if possible
    feature_names = list(num_cols)
    if cat_cols:
        try:
            ohe_obj = model_pipeline.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
            cat_ohe_names = list(ohe_obj.get_feature_names_out(cat_cols))
            feature_names += cat_ohe_names
        except Exception:
            # as final fallback, append raw cat column names
            feature_names += cat_cols

with open('models/feature_names.txt', 'w', encoding='utf-8') as f:
    for fn in feature_names:
        f.write(str(fn) + '\n')

print("\n✅ Model saved to models/hp_model_v1.pkl")
print("✅ Feature names saved to models/feature_names.txt")
print("✅ Sample test data saved to models/X_test_sample.csv")
