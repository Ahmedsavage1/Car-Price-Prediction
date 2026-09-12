import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------
# Load model + preprocessing artifacts (all computed from X_train only, during training)
# ---------------------------------------------
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
rare_brands = joblib.load('models/rare_brands.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
dropdown_options = joblib.load('models/dropdown_options.pkl')

# Fixed real-world mapping, same one used in the notebooks (not a learned statistic)
fuel_to_unit = {'CNG': 'km/kg', 'LPG': 'km/kg', 'Diesel': 'kmpl', 'Petrol': 'kmpl'}

# Columns one-hot encoded during training (same list used in 02_data_preprocessing.ipynb)
onehot_cols = ['Transmission', 'Mileage_Unit', 'Fuel_Type', 'Location', 'Brand']

# Fixed ordinal mapping, same one used in the notebooks
owner_order = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}

# Numeric columns that were scaled during training (Seats was kept unscaled, on purpose)
numeric_cols = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Car_Age']

st.title("Used Car Price Predictor")
st.write("Fill in the car's details to get an estimated price.")

# ---------------------------------------------
# User input
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    brand_choice = st.selectbox("Brand", dropdown_options['Brand'] + ['Other / Not listed'])
    location = st.selectbox("Location", dropdown_options['Location'])
    fuel_type = st.selectbox("Fuel Type", dropdown_options['Fuel_Type'])
    transmission = st.selectbox("Transmission", dropdown_options['Transmission'])
    owner_type = st.selectbox("Owner Type", dropdown_options['Owner_Type'])
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)

with col2:
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5, step=1)
    engine = st.number_input("Engine (CC)", min_value=500.0, value=1200.0, step=50.0)
    power = st.number_input("Power (bhp)", min_value=30.0, value=90.0, step=5.0)
    mileage = st.number_input("Mileage", min_value=0.0, value=18.0, step=0.5)

# Mileage_Unit is derived automatically from Fuel_Type, not asked from the user
# (same fixed mapping used in the notebooks — the user never sees or picks this)
mileage_unit = fuel_to_unit[fuel_type]

# Rare brands were grouped into "Other" during training — apply the same rule here
brand = brand_choice if brand_choice in dropdown_options['Brand'] else 'Other'

# ---------------------------------------------
# Predict
# ---------------------------------------------
if st.button("Predict Price"):

    # Build a single-row DataFrame matching the raw column shape before encoding
    input_df = pd.DataFrame([{
        'Location': location,
        'Kilometers_Driven': kilometers_driven,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Seats': seats,
        'Mileage_Unit': mileage_unit,
        'Car_Age': car_age,
        'Brand': brand
    }])

    # One-hot encode, then align columns to exactly match training (missing cols -> 0)
    input_encoded = pd.get_dummies(input_df, columns=onehot_cols, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Convert any bool columns from get_dummies into 0/1 (same as training)
    bool_cols = input_encoded.select_dtypes(include='bool').columns
    input_encoded[bool_cols] = input_encoded[bool_cols].astype(int)

    # Ordinal encode Owner_Type (same fixed mapping as training)
    input_encoded['Owner_Type'] = pd.Series([owner_type]).map(owner_order).iloc[0]

    # Scale numeric columns using the scaler fit on X_train (no new fitting here)
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    # Predict
    predicted_price = model.predict(input_encoded)[0]

    st.success(f"Estimated Price: {predicted_price:.2f} Lakhs")