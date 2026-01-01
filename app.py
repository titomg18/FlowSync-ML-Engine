from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "kmeans_transport.pkl"))

class TransportInput(BaseModel):
    distance: float
    frequency: float
    cost: float

@app.post("/predict")
def predict(data: TransportInput):
    X = np.array([[data.distance, data.frequency, data.cost]])
    cluster = int(model.predict(X)[0])
    return {"cluster": cluster}
