import os
import pickle
import numpy as np
import pandas as pd
from src.utils.file_ops import load_object
from src.utils.exceptions import CustomException
from src.utils.logger import logger
import sys

MODELS_DIR = "data/models"
PREPROCESSOR_PATH = "data/transformed_data/preprocessor.pkl"

class HousePricePredictor:
    def __init__(self, purpose: str):
        self.purpose = purpose.lower()
        if self.purpose not in ['for sale','for rent']:
            raise ValueError("Purpose must be either 'sale' or 'rent'")
        self.model_path = os.path.join(MODELS_DIR,f"{self.purpose}_model.pkl")
        self.preprocessor = load_object(PREPROCESSOR_PATH)
        self.model = load_object(self.model_path)

        logger.info(f"{self.purpose.capitalize()} model loaded successfully.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        try:
            X_processed = self.preprocessor.transform(df)
            preds = self.model.predict(X_processed)

            return preds
        except Exception as e:
            raise CustomException(f"Prediction failed: {e}",sys)
        
if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "bedrooms":[3],
        "baths":[2],
        "Area_in_Marla":[5],
        "location": ["Gulberg"],
        'property_type': ["House"],
        "purpose" : ["For Sale"]
    })

    predictor = HousePricePredictor("for sale")
    print("Predicted Price:", predictor.predict(sample_df))


