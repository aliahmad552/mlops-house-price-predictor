# Auto-generated placeholder
# Replace or extend this file with your project-specific logic.
from pydantic import BaseModel

class HouseFeatures(BaseModel):
    Area_in_Marla: float
    baths: int
    bedrooms: int
    city: str
    location: str
    property_type: str
    purpose: str
