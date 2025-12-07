from fastapi import FastAPI
from src.api.schemas import HouseFeatures
from src.api.predictor import make_prediction, df

app = FastAPI(
    title="Pakistan House Price Predictor API",
    description="Predict Sale or Rent prices using separate ML models",
    version="1.0"
)



@app.get("/")
def home():
    return {"message": "House Price Predictor API Running 🚀"}

@app.get("/cities")
def get_cities():
    cities = df['city'].unique().tolist()
    return {"cities": cities}

@app.get("/locations/{city_name}")
def get_locations(city_name: str):

    # Filter locations based on selected city
    filtered_locations = df[df['city'] == city_name]['location'].unique().tolist()

    if not filtered_locations:
        return {"error": f"No locations found for city: {city_name}"}

    return {
        "city": city_name,
        "locations": filtered_locations
    }

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Predict house price based on user input
    """
    try:
        prediction = make_prediction(features)
        return {
            "purpose": features.purpose,
            "predicted_price": prediction
        }
    except ValueError as e:
        return {"error": str(e)}
