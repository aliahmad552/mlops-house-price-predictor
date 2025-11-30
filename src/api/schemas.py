from pydantic import BaseModel

class HouseFeatures(BaseModel):
    location: str
    city: str
    property_type: str
    purpose: str  # "sale" or "rent"
    bedrooms: int
    baths: int
    Area_in_Marla: float