from fastapi import FastAPI
from src.api.schemas import HouseFeatures
from src.api.predictor import make_prediction


app = FastAPI(
    title="Pakistan House Price Predictor API",
    description="Predict Sale or Rent prices using MLflow Models",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "House Price Predictor API Running 🚀"}


@app.post("/predict/{purpose}")
def predict_price(purpose: str, features: HouseFeatures):
    """
    purpose: sale / rent
    """
    if purpose not in ["for sale", "for rent"]:
        return {"error": "purpose must be sale or rent"}

    prediction = make_prediction(features, purpose)
    return {
        "purpose": purpose,
        "predicted_price": prediction
    }
