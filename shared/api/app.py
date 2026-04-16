from fastapi import FastAPI
import numpy as np
import pickle
import tensorflow as tf
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="Smart Home API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'lstm_model.keras'))
with open(os.path.join(BASE_DIR, 'data', 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

class SensorEvent(BaseModel):
    sensor: str
    hour: int
    minute: int
    weekday: int
    is_weekend: int

class PredictionRequest(BaseModel):
    events: List[SensorEvent]

@app.get("/")
def root():
    return {"status": "Smart Home API работает!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    sequence = []
    for event in request.events[-10:]:
        try:
            sensor_id = le.transform([event.sensor])[0]
        except:
            sensor_id = 0
        sequence.append([
            sensor_id,
            event.hour,
            event.minute,
            event.weekday,
            event.is_weekend
        ])
    
    X = np.array([sequence])
    pred = model.predict(X, verbose=0)
    sensor_id = pred.argmax()
    confidence = float(pred[0][sensor_id])
    
    return {
        "predicted_sensor": le.classes_[sensor_id],
        "confidence": round(confidence, 3),
        "all_probabilities": {
            le.classes_[i]: round(float(pred[0][i]), 3)
            for i in range(len(le.classes_))
        }
    }

@app.get("/sensors")
def get_sensors():
    return {"sensors": list(le.classes_)}