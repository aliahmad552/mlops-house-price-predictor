import numpy as np
import pandas as pd
import joblib
from src.api.mlflow_loader import sale_model, rent_model

# Load preprocessors
sale_preprocessor = joblib.load("artifacts/transformed_data/sale/preprocessor.pkl")
rent_preprocessor = joblib.load("artifacts/transformed_data/rent/preprocessor.pkl")


def make_prediction(data, purpose: str):
   
    model = sale_model if purpose == "sale" else rent_model
    preprocessor = sale_preprocessor if purpose == "sale" else rent_preprocessor

    # Convert input to DataFrame (preserves column names for preprocessor)
    input_df = pd.DataFrame([{
        "location": data.location,
        "city": data.city,
        "property_type": data.property_type,
        "baths": data.baths,
        "bedrooms": data.bedrooms,
        "Area_in_Marla": data.Area_in_Marla
    }])

    # Apply preprocessing
    features = preprocessor.transform(input_df)

    # Predict using the MLflow model
    prediction = model.predict(features)

    return float(prediction[0])