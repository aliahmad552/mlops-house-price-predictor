import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.utils.exceptions import CustomException
from src.utils.file_ops import save_object
from sklearn.metrics import r2_score
from src.utils.logger import logger

TRANSFORMED_DIR = 'data/transformed_data'
MODELS_DIR = 'data/models'
os.makedirs(MODELS_DIR, exist_ok = True)

def train_sale_model():
    try:
        X_train = np.load(os.path.join(TRANSFORMED_DIR,'X_train.npy'))
        X_test = np.load(os.path.join(TRANSFORMED_DIR,'X_test.npy'))
        y_train = np.load(os.path.join(TRANSFORMED_DIR,'y_train.npy'))
        y_test = np.load(os.path.join(TRANSFORMED_DIR,'y_test.npy'))

        model = RandomForestRegressor(n_estimators=140,max_depth = 10,random_state = 42, n_jobs = -1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test,y_pred)
        logger.info(f"Sale Model Tranined | R2 Score: {r2:.4f}")

        save_object(model,os.path.join(MODELS_DIR,"sale_model.pkl"))
        logger.info("Sale model saved successfully.")

    except Exception as e:
        raise CustomException(f"Sale Model Training Failed: {e}",sys)
    
if __name__ == "__main__":
    train_sale_model()