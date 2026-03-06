import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="🏠 California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --------------------------
# Load saved models & scaler
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    linear_model = pickle.load(open(os.path.join(BASE_DIR, "linear_model.pkl"), "rb"))
    ridge_model   = pickle.load(open(os.path.join(BASE_DIR, "ridge_model.pkl"),  "rb"))
    lasso_model   = pickle.load(open(os.path.join(BASE_DIR, "lasso_model.pkl"),  "rb"))
    scaler        = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"),        "rb"))
    return linear_model, ridge_model, lasso_model, scaler

linear_model, ridge_model, lasso_model, scaler = load_models()

# --------------------------
# App UI
# --------------------------
st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Enter house features below and select a regression model to predict the **median house value**."
)

st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Regression Model",
    ["Linear Regression", "Ridge Regression", "Lasso Regression"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Models trained on the California Housing Dataset.**\n\n"
    "Ridge & Lasso use GridSearchCV-optimized alpha values."
)

# --------------------------
# Input Features
# --------------------------
st.subheader("📋 House Features")

col1, col2 = st.columns(2)

with col1:
    longitude          = st.number_input("Longitude",           value=-118.25, format="%.4f")
    housing_median_age = st.number_input("Housing Median Age",  value=29,      min_value=1, max_value=52)
    total_bedrooms     = st.number_input("Total Bedrooms",       value=435,     min_value=1)
    households         = st.number_input("Households",           value=409,     min_value=1)

with col2:
    latitude    = st.number_input("Latitude",      value=34.05, format="%.4f")
    total_rooms = st.number_input("Total Rooms",   value=2127,  min_value=1)
    population  = st.number_input("Population",    value=1166,  min_value=1)
    median_income = st.number_input("Median Income (in $10k units)", value=3.87, min_value=0.0, format="%.2f")

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# --------------------------
# Feature Engineering
# --------------------------
rooms_per_household      = total_rooms / households
bedrooms_per_room        = total_bedrooms / total_rooms
population_per_household = population / households

# One-hot encoding (order must match training)
ocean_categories = ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"]
ocean_input = [1 if ocean_proximity == cat else 0 for cat in ocean_categories]

# Combine all features
X_input = np.array([
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income,
    rooms_per_household, bedrooms_per_room, population_per_household,
    *ocean_input
]).reshape(1, -1)

# Scale features
X_input_scaled = scaler.transform(X_input)

# --------------------------
# Predict
# --------------------------
st.markdown("---")

if st.button("🔍 Predict House Price", use_container_width=True):
    if model_choice == "Linear Regression":
        price = linear_model.predict(X_input_scaled)[0]
    elif model_choice == "Ridge Regression":
        price = ridge_model.predict(X_input_scaled)[0]
    else:
        price = lasso_model.predict(X_input_scaled)[0]

    st.success(f"💰 **Predicted Median House Value: ${price:,.2f}**")

    # Show derived features
    with st.expander("🔎 View Derived Features Used"):
        st.write({
            "Rooms per Household":       round(rooms_per_household, 3),
            "Bedrooms per Room":         round(bedrooms_per_room, 3),
            "Population per Household":  round(population_per_household, 3),
            "Ocean Proximity Encoding":  dict(zip(ocean_categories, ocean_input))
        })

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn | California Housing Dataset")
