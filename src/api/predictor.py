import numpy as np
import pandas as pd
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/api

ARTIFACTS_DIR = os.path.join(BASE_DIR, "../../artifacts")

# Load preprocessors
sale_preprocessor = joblib.load("artifacts/transformed_data/sale/preprocessor.pkl")
rent_preprocessor = joblib.load("artifacts/transformed_data/rent/preprocessor.pkl")

sale_model = joblib.load("artifacts/models/xgb_sale.pkl")
rent_model = joblib.load("artifacts/models/xgb_rent.pkl")

df = pd.read_csv("artifacts/raw_data/data.csv")

def make_prediction(data):
    """
    data: HouseFeatures object
    Returns: predicted price (float)
    """

    # purpose is read from input
    purpose = data.purpose.lower()
    if purpose in ["for sale", "sale"]:
        model = sale_model
        preprocessor = sale_preprocessor
    elif purpose in ["for rent", "rent"]:
        model = rent_model
        preprocessor = rent_preprocessor
    else:
        raise ValueError("Invalid purpose. Must be 'for sale' or 'for rent'")

    input_df = pd.DataFrame([{
        "location": data.location,
        "city": data.city,
        "property_type": data.property_type,
        "purpose": data.purpose,  # include for preprocessor
        "baths": data.baths,
        "bedrooms": data.bedrooms,
        "Area_in_Marla": data.Area_in_Marla
    }])

    features = preprocessor.transform(input_df)
    prediction = model.predict(features)
    return float(prediction[0])
