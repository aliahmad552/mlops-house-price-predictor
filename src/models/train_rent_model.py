import os
import sys
import yaml
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.utils.file_ops import save_object


class TrainRentConfig:
    def __init__(self):
        with open("src/config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        rent_cfg = cfg["training"]["rent"]

        self.transformed_dir = rent_cfg["transformed_dir"]
        self.model_path = rent_cfg["model_path"]
        self.mlflow_experiment_path = rent_cfg["mlflow_experiment"]

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        self.params = params["model"]["rent"]


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

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return rmse, r2

    def run(self):

        # MLflow setup
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(self.config.mlflow_experiment_path)

        X_train, X_test, y_train, y_test = self.load_data()

        with mlflow.start_run():

            # ---------------- XGBoost Model ---------------- #
            model = XGBRegressor(
                n_estimators=self.config.params["n_estimators"],
                learning_rate=self.config.params["learning_rate"],
                max_depth=self.config.params["max_depth"],
                subsample=self.config.params["subsample"],
                colsample_bytree=self.config.params["colsample_bytree"],
                random_state=self.config.params["random_state"]
            )

            logger.info("Training XGBoost model…")
            model.fit(X_train, y_train)

            # Evaluate
            rmse, r2 = self.evaluate(model, X_test, y_test)
            logger.info(f"XGBoost → RMSE: {rmse}, R2: {r2}")

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Log model parameters
            for param_name, value in model.get_params().items():
                mlflow.log_param(param_name, value)

            # -------- Save best model locally for DVC -------- #
            save_object(model, self.config.model_path)
            logger.info("BEST RENT MODEL SAVED LOCALLY (XGBoost)")

            # -------- Log model to MLflow -------- #
            mlflow.xgboost.log_model(model, artifact_path=self.config.model_path)

            # -------- Register model in MLflow Model Registry -------- #
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/{self.config.model_path}",
                name="RentPriceModel"
            )

            logger.info("✔ XGBoost RENT MODEL LOGGED & REGISTERED SUCCESSFULLY")


if __name__ == "__main__":
    TrainRentModel().run()
