from fastapi import FastAPI

from pydantic import BaseModel
import joblib

app = FastAPI()

clf = joblib.load("model/classifier.pkl")
embedder = joblib.load("model/embedder.pkl")

class TweetRequest(BaseModel):
    text: str

@app.get("/")

def root() :
    return {"status": "ok"}

@app.post("/predict")

def predict(request: TweetRequest):
    embedding = embedder.encode([request.text])
    prob =clf.predict_proba(embedding)[0][1]
    label = "SCAM" if prob > 0.5 else "LEGIT"
    return {
        "text": request.text,
        "scam_probability": round(float(prob), 4) ,
        "label": label
    }
