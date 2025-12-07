import os
from dotenv import load_dotenv
import mlflow
import xgboost as xgb

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# def load_model_from_run(run_id: str, artifact_path: str):
#     """
#     Load a model directly from a specific MLflow run's artifact path.
    
#     Args:
#         run_id (str): The MLflow run ID where the model is logged
#         artifact_path (str): Path inside the run artifacts (e.g., 'xgb_rent_model')

#     Returns:
#         PyFunc model
#     """
#     model_uri = f"runs:/{run_id}/{artifact_path}"
#     print(f"[INFO] Loading model from run artifact: {model_uri}")
#     return mlflow.pyfunc.load_model(model_uri)


# Example usage: supply your run IDs for Sale & Rent models
# sale_run_id = "e143091362ac4e3da620ca2712529e71"
# rent_run_id = "02d59815c5f543e394a077d64153158c"

# sale_model = load_model_from_run(sale_run_id, "artifacts/models/xgb_sale")
# rent_model = load_model_from_run(rent_run_id, "artifacts/models/xgb_rent")
