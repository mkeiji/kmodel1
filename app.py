from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

app = FastAPI(title="Sentiment Analysis API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    X = vectorizer.transform([review.text])
    prediction = model.predict(X)[0]
    return {"review": review.text, "sentiment": prediction}
