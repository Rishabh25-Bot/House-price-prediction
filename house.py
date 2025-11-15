# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(layout='wide', page_title="House Price Predictor")

@st.cache_data
def load_data_local():
    if os.path.exists('new_dataset.xlsx'):
        return pd.read_excel('new_dataset.xlsx', engine='openpyxl')
    elif os.path.exists('new_dataset.csv'):
        return pd.read_csv('new_dataset.csv')
    else:
        return None

@st.cache_resource
def load_model(path='models/hp_model_v1.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    return None

st.title("House Price Predictor — improved")

# Sidebar: upload or use local
st.sidebar.header("Data / Model")
uploaded = st.sidebar.file_uploader("Upload dataset (.csv or .xlsx)", type=['csv','xlsx'])
if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine='openpyxl')
else:
    df = load_data_local()
    if df is None:
        st.sidebar.warning("No dataset found. Upload a file or place new_dataset.xlsx in the app folder.")
        st.stop()

model = load_model()
if model is None:
    st.sidebar.error("Model not found. Run train_model.py to create models/hp_model_v1.pkl")
    st.stop()

# Build a quick UI based on available columns
cols = list(df.columns)
st.sidebar.write("Columns detected:", cols)

# Attempt to detect common fields (modify if your columns are named differently)
area_col = 'Area_sqft' if 'Area_sqft' in cols else st.sidebar.selectbox("Area column", options=cols, index=0)
locality_col = 'Locality' if 'Locality' in cols else None
if 'Locality' in cols:
    locality_col = 'Locality'
else:
    # fallback: choose first categorical-like column
    cat_candidates = [c for c in cols if df[c].dtype == 'object']
    if cat_candidates:
        locality_col = st.sidebar.selectbox("Select locality-like column", options=cat_candidates)

# Prediction form
st.header("Make a prediction")
with st.form("predict_form"):
    left, right = st.columns(2)
    with left:
        area = st.number_input("Area (sqft)", value=int(df[area_col].median() if area_col in df else 1000))
        bhk = st.number_input("BHK (if available)", value=int(df['BHK'].median()) if 'BHK' in df.columns else 2)
        baths = st.number_input("Bathrooms (if available)", value=int(df['bathrooms'].median()) if 'bathrooms' in df.columns else 2)
    with right:
        locality = None
        if locality_col:
            locality = st.selectbox("Locality", options=sorted(df[locality_col].dropna().unique()))
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        # create input DF matching training features (best-effort)
        input_dict = {}
        # numeric columns — try to find them from model preprocessor
        # Minimal: set what's available
        if area_col:
            input_dict[area_col] = area
        if 'BHK' in df.columns:
            input_dict['BHK'] = bhk
        if 'bathrooms' in df.columns:
            input_dict['bathrooms'] = baths
        if locality_col and locality is not None:
            input_dict[locality_col] = locality
        X_new = pd.DataFrame([input_dict])

        # Predict
        try:
            pred_per_sqft = model.predict(X_new)[0]
            total_price = pred_per_sqft * area
            st.metric("Predicted price per sqft (INR)", f"₹{pred_per_sqft:,.0f}")
            st.metric("Predicted total price (INR)", f"₹{total_price:,.0f}")
        except Exception as e:
            st.error("Prediction failed. The model expects certain columns. See sidebar for columns.")
            st.exception(e)

# Feature importance (basic)
st.header("Model Feature Importance (basic)")
# try to load names
feat_file = 'models/feature_names.txt'
if os.path.exists(feat_file):
    with open(feat_file, 'r', encoding='utf8') as f:
        feat_names = [l.strip() for l in f if l.strip()]
else:
    feat_names = None

try:
    import numpy as np
    reg = model.named_steps['reg']
    importances = reg.feature_importances_
    if feat_names and len(feat_names) == len(importances):
        fig, ax = plt.subplots(figsize=(8,4))
        idx = np.argsort(importances)[-15:]
        ax.barh([feat_names[i] for i in idx], importances[idx])
        ax.set_title("Top 15 features")
        st.pyplot(fig)
    else:
        st.write("Feature importance available but feature names not found or mismatch.")
        st.write(importances[:30])
except Exception as e:
    st.write("Couldn't compute feature importance.")
    st.exception(e)
