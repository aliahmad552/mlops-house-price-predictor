# Auto-generated placeholder
# Replace or extend this file with your project-specific logic.
# Placeholder: shared prediction utilities (load model, preprocess input)
from src.utils.logger import logger

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)
