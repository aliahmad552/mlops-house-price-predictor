import os
import sys
import yaml
import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.utils.file_ops import save_object


class TrainSaleConfig:
    def __init__(self):
        # Load paths from config.yaml
        with open("src/config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        sale_cfg = cfg["training"]["sale"]
        self.transformed_dir = sale_cfg["transformed_dir"]
        self.model_path = sale_cfg["model_path"]
        self.preprocessor_path = sale_cfg["preprocessor_path"]
        self.mlflow_experiment = sale_cfg["mlflow_experiment"]

        # Load hyperparameters from params.yaml
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        sale_params = params["model"]["sale"]
        self.n_estimators = sale_params["n_estimators"]
        self.learning_rate = sale_params["learning_rate"]
        self.max_depth = sale_params["max_depth"]
        self.subsample = sale_params["subsample"]
        self.colsample_bytree = sale_params["colsample_bytree"]
        self.random_state = sale_params["random_state"]


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

    def train(self, X_train, y_train):
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
        with mlflow.start_run():
            model = self.train(X_train, y_train)
            rmse, r2 = self.evaluate(model, X_test, y_test)

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            mlflow.xgboost.log_model(model, "xgb_sale_model")

            save_object(model, self.config.model_path)
            logger.info(f"Model saved at: {self.config.model_path}")


if __name__ == "__main__":
    TrainSaleModel().run()
