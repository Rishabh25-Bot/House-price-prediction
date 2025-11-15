"""
House Price Prediction - Complete Python script
Replace file paths and tweak parameters as needed.
Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
Optional: geopandas (for geospatial features), gspread (for Google Sheets export)

This is a single-file runnable script that:
 - Loads a CSV dataset (e.g., Kaggle Ames or Boston-like CSV)
 - Cleans data, imputes missing values
 - Encodes categorical variables
 - Engineers features (age, total baths, price_per_sqft etc.)
 - Splits data, trains LinearRegression, RandomForest, XGBoost
 - Evaluates with RMSE, MAE, R2, MAPE
 - Saves the best model
 - Provides a prediction helper function

Note: For a production-ready pipeline, convert parts into modular functions and add proper logging.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Optional: if xgboost is available
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ------------------------
# Configuration
# ------------------------
DATA_PATH = 'train.csv'  # replace with your dataset path
TARGET = 'SalePrice'     # change if different
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUTPUT_DIR = 'models'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ------------------------
# Utility functions
# ------------------------

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
    return df


def evaluate_regression(true, pred):
    rmse = mean_squared_error(true, pred, squared=False)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE(%)': mape}

# ------------------------
# Basic feature engineering
# ------------------------

def basic_feature_engineering(df):
    # Assumes typical house dataset columns; adjust to your dataset.
    df = df.copy()

    # Year-based features
    current_year = pd.Timestamp.now().year
    if 'YearBuilt' in df.columns:
        df['House_Age'] = current_year - df['YearBuilt']
    if 'YearRemodAdd' in df.columns:
        df['Years_Since_Remodel'] = current_year - df['YearRemodAdd']

    # Bathrooms
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['Total_Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

    # Total square footage
    sqf_cols = [c for c in ['GrLivArea','TotalBsmtSF','1stFlrSF','2ndFlrSF'] if c in df.columns]
    if sqf_cols:
        df['Total_Sqft'] = df[sqf_cols].sum(axis=1)

    # Price per sqft (if target exists and sqft exists)
    if TARGET in df.columns and 'Total_Sqft' in df.columns:
        # avoid division by zero
        df['Price_per_Sqft'] = df[TARGET] / df['Total_Sqft'].replace(0, np.nan)

    # Simplify some categorical levels if present
    if 'Neighborhood' in df.columns:
        # keep top 15 neighborhoods, rest as 'Other'
        topn = df['Neighborhood'].value_counts().nlargest(15).index
        df['Neighborhood_grp'] = df['Neighborhood'].where(df['Neighborhood'].isin(topn), other='Other')

    return df

# ------------------------
# Preprocessing and pipeline
# ------------------------

def build_preprocessing(df, drop_cols=None):
    # Identify numeric and categorical columns
    if drop_cols is None:
        drop_cols = []
    features = [c for c in df.columns if c != TARGET and c not in drop_cols]

    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Simple imputers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor, features

# ------------------------
# Model training helpers
# ------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    # Linear Regression baseline
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    results['LinearRegression'] = evaluate_regression(y_test, y_pred)
    joblib.dump(lr_pipeline, os.path.join(MODEL_OUTPUT_DIR, 'linear_regression.pkl'))

    # Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    rf_params = {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [None, 10, 20]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    results['RandomForest'] = evaluate_regression(y_test, y_pred_rf)
    joblib.dump(best_rf, os.path.join(MODEL_OUTPUT_DIR, 'random_forest.pkl'))

    # XGBoost (if available)
    if HAS_XGB:
        xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1))])
        xgb_params = {
            'regressor__n_estimators': [100],
            'regressor__max_depth': [3,6],
            'regressor__learning_rate': [0.05, 0.1]
        }
        xg_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        xg_grid.fit(X_train, y_train)
        best_xg = xg_grid.best_estimator_
        y_pred_xg = best_xg.predict(X_test)
        results['XGBoost'] = evaluate_regression(y_test, y_pred_xg)
        joblib.dump(best_xg, os.path.join(MODEL_OUTPUT_DIR, 'xgboost.pkl'))

    return results

# ------------------------
# Prediction helper
# ------------------------

def load_model_and_predict(model_path, input_df):
    model = joblib.load(model_path)
    preds = model.predict(input_df)
    return preds

# ------------------------
# Main execution
# ------------------------
if _name_ == '__main__':
    # 1) Load
    df = load_data(DATA_PATH)

    # quick sanity check: ensure target exists
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset. Update TARGET or dataset.")

    # 2) Feature engineering
    df_fe = basic_feature_engineering(df)

    # 3) Choose columns to drop (IDs, text-heavy columns)
    drop_cols = [c for c in ['Id','PID','Order'] if c in df_fe.columns]

    # 4) Build preprocessor
    preprocessor, features = build_preprocessing(df_fe, drop_cols=drop_cols)

    # 5) Train/test split
    X = df_fe[features]
    y = df_fe[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 6) Train and evaluate
    print('Training models (this may take a while for ensemble models)...')
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

    # 7) Print results
    print('\nEvaluation results:')
    for model_name, metrics in results.items():
        print(f"--- {model_name} ---")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}")

    # 8) Save a simple feature importance example for RandomForest if available
    rf_model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest.pkl')
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
        # to get feature names after OneHotEncoding, we need to extract transformer feature names
        try:
            pre = rf.named_steps['preprocessor']
            # numeric features
            num_cols = pre.transformers_[0][2]
            # get onehot feature names
            ohe = pre.transformers_[1][1].named_steps['onehot']
            cat_cols = pre.transformers_[1][2]
            ohe_names = ohe.get_feature_names_out(cat_cols)
            feature_names = list(num_cols) + list(ohe_names)
            importances = rf.named_steps['regressor'].feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
            feat_imp.to_csv(os.path.join(MODEL_OUTPUT_DIR, 'feature_importances_top30.csv'))
            print(f"Saved top feature importances to {os.path.join(MODEL_OUTPUT_DIR, 'feature_importances_top30.csv')}")
        except Exception as e:
            print('Could not extract feature importances automatically:', e)

    print('All done. Models saved in the models/ directory.')

# ------------------------
# Optional: Example of exporting predictions to Google Sheets (commented)
# ------------------------
#
# To use Google Sheets export, install `gspread` and `gspread_dataframe`, create a service account,
# download credentials JSON and set the path. This is optional and commented out by default.
#
# from gspread_dataframe import set_with_dataframe
# import gspread
#
# SERVICE_ACCOUNT_FILE = 'service_account.json'
# SHEET_NAME = 'HousePricePredictions'
#
# gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
# sh = gc.create(SHEET_NAME)
# worksheet = sh.sheet1
#
# sample_preds = pd.DataFrame({'Id': X_test.index, 'TruePrice': y_test, 'Pred_RF': joblib.load(rf_model_path).predict(X_test)})
# set_with_dataframe(worksheet, sample_preds)
#
# ------------------------
# Tips & Notes:
# - Replace DATA_PATH with your CSV file; if your dataset uses different column names adjust engineering steps.
# - If model accuracy is poor: engineer better location features, incorporate distance to city center/schools,
#   try log-transforming SalePrice, or use target encoding for high-cardinality categoricals.
# - For production: wrap preprocessing & model into a single Pipeline and serve via FastAPI/Streamlit.
# ------------------------
"""
House Price Prediction - Complete Python script
Replace file paths and tweak parameters as needed.
Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
Optional: geopandas (for geospatial features), gspread (for Google Sheets export)

This is a single-file runnable script that:
 - Loads a CSV dataset (e.g., Kaggle Ames or Boston-like CSV)
 - Cleans data, imputes missing values
 - Encodes categorical variables
 - Engineers features (age, total baths, price_per_sqft etc.)
 - Splits data, trains LinearRegression, RandomForest, XGBoost
 - Evaluates with RMSE, MAE, R2, MAPE
 - Saves the best model
 - Provides a prediction helper function

Note: For a production-ready pipeline, convert parts into modular functions and add proper logging.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Optional: if xgboost is available
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ------------------------
# Configuration
# ------------------------
DATA_PATH = 'train.csv'  # replace with your dataset path
TARGET = 'SalePrice'     # change if different
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUTPUT_DIR = 'models'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ------------------------
# Utility functions
# ------------------------

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
    return df


def evaluate_regression(true, pred):
    rmse = mean_squared_error(true, pred, squared=False)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE(%)': mape}

# ------------------------
# Basic feature engineering
# ------------------------

def basic_feature_engineering(df):
    # Assumes typical house dataset columns; adjust to your dataset.
    df = df.copy()

    # Year-based features
    current_year = pd.Timestamp.now().year
    if 'YearBuilt' in df.columns:
        df['House_Age'] = current_year - df['YearBuilt']
    if 'YearRemodAdd' in df.columns:
        df['Years_Since_Remodel'] = current_year - df['YearRemodAdd']

    # Bathrooms
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['Total_Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

    # Total square footage
    sqf_cols = [c for c in ['GrLivArea','TotalBsmtSF','1stFlrSF','2ndFlrSF'] if c in df.columns]
    if sqf_cols:
        df['Total_Sqft'] = df[sqf_cols].sum(axis=1)

    # Price per sqft (if target exists and sqft exists)
    if TARGET in df.columns and 'Total_Sqft' in df.columns:
        # avoid division by zero
        df['Price_per_Sqft'] = df[TARGET] / df['Total_Sqft'].replace(0, np.nan)

    # Simplify some categorical levels if present
    if 'Neighborhood' in df.columns:
        # keep top 15 neighborhoods, rest as 'Other'
        topn = df['Neighborhood'].value_counts().nlargest(15).index
        df['Neighborhood_grp'] = df['Neighborhood'].where(df['Neighborhood'].isin(topn), other='Other')

    return df

# ------------------------
# Preprocessing and pipeline
# ------------------------

def build_preprocessing(df, drop_cols=None):
    # Identify numeric and categorical columns
    if drop_cols is None:
        drop_cols = []
    features = [c for c in df.columns if c != TARGET and c not in drop_cols]

    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Simple imputers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor, features

# ------------------------
# Model training helpers
# ------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    # Linear Regression baseline
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    results['LinearRegression'] = evaluate_regression(y_test, y_pred)
    joblib.dump(lr_pipeline, os.path.join(MODEL_OUTPUT_DIR, 'linear_regression.pkl'))

    # Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    rf_params = {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [None, 10, 20]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    results['RandomForest'] = evaluate_regression(y_test, y_pred_rf)
    joblib.dump(best_rf, os.path.join(MODEL_OUTPUT_DIR, 'random_forest.pkl'))

    # XGBoost (if available)
    if HAS_XGB:
        xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1))])
        xgb_params = {
            'regressor__n_estimators': [100],
            'regressor__max_depth': [3,6],
            'regressor__learning_rate': [0.05, 0.1]
        }
        xg_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        xg_grid.fit(X_train, y_train)
        best_xg = xg_grid.best_estimator_
        y_pred_xg = best_xg.predict(X_test)
        results['XGBoost'] = evaluate_regression(y_test, y_pred_xg)
        joblib.dump(best_xg, os.path.join(MODEL_OUTPUT_DIR, 'xgboost.pkl'))

    return results

# ------------------------
# Prediction helper
# ------------------------

def load_model_and_predict(model_path, input_df):
    model = joblib.load(model_path)
    preds = model.predict(input_df)
    return preds

# ------------------------
# Main execution
# ------------------------
# ------------------------
# Main execution
# ------------------------
if __name__ == '__main__':
    # 1) Load
    df = load_data(DATA_PATH)

    # quick sanity check: ensure target exists
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset. Update TARGET or dataset.")

    # 2) Feature engineering
    df_fe = basic_feature_engineering(df)

    # 3) Choose columns to drop (IDs, text-heavy columns)
    drop_cols = [c for c in ['Id','PID','Order'] if c in df_fe.columns]

    # 4) Build preprocessor
    preprocessor, features = build_preprocessing(df_fe, drop_cols=drop_cols)

    # 5) Train/test split
    X = df_fe[features]
    y = df_fe[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 6) Train and evaluate
    print('Training models (this may take a while for ensemble models)...')
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

    # 7) Print results
    print('\nEvaluation results:')
    for model_name, metrics in results.items():
        print(f"--- {model_name} ---")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}")

    # 8) Save a simple feature importance example for RandomForest if available
    rf_model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest.pkl')
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
        # to get feature names after OneHotEncoding, we need to extract transformer feature names
        try:
            pre = rf.named_steps['preprocessor']
            # numeric features
            num_cols = pre.transformers_[0][2]
            # get onehot feature names
            ohe = pre.transformers_[1][1].named_steps['onehot']
            cat_cols = pre.transformers_[1][2]
            ohe_names = ohe.get_feature_names_out(cat_cols)
            feature_names = list(num_cols) + list(ohe_names)
            importances = rf.named_steps['regressor'].feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
            feat_imp.to_csv(os.path.join(MODEL_OUTPUT_DIR, 'feature_importances_top30.csv'))
            print(f"Saved top feature importances to {os.path.join(MODEL_OUTPUT_DIR, 'feature_importances_top30.csv')}")
        except Exception as e:
            print('Could not extract feature importances automatically:', e)

    print('All done. Models saved in the models/ directory.')

# ------------------------
# Optional: Example of exporting predictions to Google Sheets (commented)
# ------------------------
#
# To use Google Sheets export, install `gspread` and `gspread_dataframe`, create a service account,
# download credentials JSON and set the path. This is optional and commented out by default.
#
# from gspread_dataframe import set_with_dataframe
# import gspread
#
# SERVICE_ACCOUNT_FILE = 'service_account.json'
# SHEET_NAME = 'HousePricePredictions'
#
# gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
# sh = gc.create(SHEET_NAME)
# worksheet = sh.sheet1
#
# sample_preds = pd.DataFrame({'Id': X_test.index, 'TruePrice': y_test, 'Pred_RF': joblib.load(rf_model_path).predict(X_test)})
# set_with_dataframe(worksheet, sample_preds)
#
# ------------------------
# Tips & Notes:
# - Replace DATA_PATH with your CSV file; if your dataset uses different column names adjust engineering steps.
# - If model accuracy is poor: engineer better location features, incorporate distance to city center/schools,
#   try log-transforming SalePrice, or use target encoding for high-cardinality categoricals.
# - For production: wrap preprocessing & model into a single Pipeline and serve via FastAPI/Streamlit.
# ------------------------
