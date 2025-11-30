import os
import sys
import yaml
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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


class TrainSaleConfig:
    def __init__(self):
        with open("src/config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        sale_cfg = cfg["training"]["sale"]

        self.transformed_dir = sale_cfg["transformed_dir"]
        self.model_path = sale_cfg["model_path"]
        self.mlflow_experiment = sale_cfg["mlflow_experiment"]

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        self.params = params["model"]["sale"]


class TrainSaleModel:
    def __init__(self):
        self.config = TrainSaleConfig()
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

    def train_model(self, model, X_train, y_train):
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

        # ---------------- MODEL DEFINITIONS ---------------- #
        models = {
            "XGBoost": XGBRegressor(
                n_estimators=self.config.params["n_estimators"],
                learning_rate=self.config.params["learning_rate"],
                max_depth=self.config.params["max_depth"],
                subsample=self.config.params["subsample"],
                colsample_bytree=self.config.params["colsample_bytree"],
                random_state=self.config.params["random_state"]
            ),

            "RandomForest": RandomForestRegressor(
                n_estimators=self.config.params["rf_n_estimators"],
                max_depth=self.config.params["rf_max_depth"],
                random_state=self.config.params["random_state"]
            ),

            "DecisionTree": DecisionTreeRegressor(
                max_depth=self.config.params["dt_max_depth"],
                random_state=self.config.params["random_state"]
            )
        }

        best_model = None
        best_rmse = float("inf")
        best_model_name = None
        best_run_id = None

        # ---------------- TRAIN EACH MODEL ---------------- #
        for model_name, model in models.items():

            with mlflow.start_run(run_name=model_name) as run:

                # Log params
                for key, value in self.config.params.items():
                    mlflow.log_param(key, value)

                model = self.train_model(model, X_train, y_train)
                rmse, r2 = self.evaluate(model, X_test, y_test)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)

                # Log model to artifacts (regular)
                if model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")

                logger.info(f"{model_name} → RMSE: {rmse}, R2: {r2}")

                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_model_name = model_name
                    best_run_id = run.info.run_id

        # ---------------- REGISTER BEST MODEL IN MLflow ---------------- #
        logger.info(f"Registering BEST model → {best_model_name}")

        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name="xgboost-sale"
        )

        # ---------------- SAVE BEST MODEL LOCALLY ---------------- #
        save_object(best_model, self.config.model_path)
        logger.info(f"BEST MODEL SAVED LOCALLY: {best_model_name} → {self.config.model_path}")


if __name__ == "__main__":
    TrainSaleModel().run()
