import os
import sys
import yaml
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.utils.file_ops import save_object
from dotenv import load_dotenv

# ---------------- Load MLflow Credentials -------------------
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class TrainRentConfig:
    def __init__(self):
        with open("src/config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        rent_cfg = cfg["training"]["rent"]

        self.transformed_dir = rent_cfg["transformed_dir"]
        self.model_path = rent_cfg["model_path"]
        self.mlflow_experiment = rent_cfg["mlflow_experiment"]

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        rent_params = params["model"]["rent"]
        self.n_estimators = rent_params["n_estimators"]
        self.learning_rate = rent_params["learning_rate"]
        self.max_depth = rent_params["max_depth"]
        self.subsample = rent_params["subsample"]
        self.colsample_bytree = rent_params["colsample_bytree"]
        self.random_state = rent_params["random_state"]


class TrainRentModel:
    def __init__(self):
        self.config = TrainRentConfig()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def load_data(self):
        try:
            X_train = np.load(f"{self.config.transformed_dir}/X_train.npy")
            X_test = np.load(f"{self.config.transformed_dir}/X_test.npy")
            y_train = np.load(f"{self.config.transformed_dir}/y_train.npy")
            y_test = np.load(f"{self.config.transformed_dir}/y_test.npy")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train):
        model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
        )
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return rmse, r2

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        mlflow.set_experiment(self.config.mlflow_experiment)

        with mlflow.start_run() as run:
            model = self.train_model(X_train, y_train)
            rmse, r2 = self.evaluate(model, X_test, y_test)

            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            # Log model artifact
            mlflow.xgboost.log_model(model, artifact_path="model")

            # Register model in MLflow Model Registry
            logger.info("Registering XGBoost Rent model in MLflow Model Registry")
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="xgboost-rent"
            )

            # Save locally
            save_object(model, self.config.model_path)
            logger.info(f"Best Rent model saved locally at: {self.config.model_path}")


if __name__ == "__main__":
    TrainRentModel().run()
