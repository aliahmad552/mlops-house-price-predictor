import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.utils.file_ops import save_object, load_object

# Paths
RAW_RENT_PATH = 'artifacts/raw_data/rent.csv'
TRANSFORMED_RENT_DIR = 'artifacts/transformed_data/rent/'

class DataTransformationRent:
    def __init__(self):
        self.raw_data_path = RAW_RENT_PATH
        self.transformed_data_dir = TRANSFORMED_RENT_DIR
        os.makedirs(self.transformed_data_dir, exist_ok=True)

    def load_data(self):
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Rent data loaded from {self.raw_data_path} with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Failed to load Rent data: {e}", sys)
        
    def split_feature_target(self, df: pd.DataFrame):
        try:
            X = df.drop(columns=['price'], axis=1)
            y = df['price']
            logger.info(f"Rent features and target separated. Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
        except Exception as e:
            raise CustomException(f"Failed to split Rent features and target: {e}", sys)
    
    def create_preprocessor(self, X: pd.DataFrame):
        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numeric_features.remove('MyUnknownColumn')
            print(categorical_features)
            print(numeric_features)


            logger.info(f"Numeric features (Rent): {numeric_features}")
            logger.info(f"Categorical features (Rent): {categorical_features}")

            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_pipeline, numeric_features),
                    ('cat', categorical_pipeline, categorical_features)
                ]
            )

            logger.info("Rent preprocessor pipeline created successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(f"Failed to create Rent preprocessor: {e}", sys)

    def run(self):
        try: 
            df = self.load_data()
            X, y = self.split_feature_target(df)

            preprocessor = self.create_preprocessor(X)
            X_processed = preprocessor.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )

            # Save transformed data
            np.save(os.path.join(self.transformed_data_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(self.transformed_data_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(self.transformed_data_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.transformed_data_dir, 'y_test.npy'), y_test)
            logger.info("Transformed Rent data saved successfully.")

            # Save preprocessor
            preprocessor_path = os.path.join(self.transformed_data_dir, 'preprocessor.pkl')
            save_object(preprocessor, preprocessor_path)
            logger.info(f"Rent preprocessor object saved at {preprocessor_path}")
        except Exception as e:
            raise CustomException(f"Rent data transformation failed: {e}", sys)


if __name__ == "__main__":
    transformer = DataTransformationRent()
    transformer.run()
