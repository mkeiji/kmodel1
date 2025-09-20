from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import joblib

class Review(BaseModel):
    text: str

# naive bayes model
nb_model = joblib.load("models/naive_bayes/sentiment_model.pkl")
vectorizer = joblib.load("models/naive_bayes/vectorizer.pkl")

# hugging face model
model_path = "models/hugging_face"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt")

# FastAPI app
app = FastAPI(title="Sentiment Analysis - Kmodel")
app.mount("/static", StaticFiles(directory="static"), name="static")

# endpoints
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/predict/naive-bayes")
def predict_sentiment(review: Review):
    X = vectorizer.transform([review.text])
    prediction = nb_model.predict(X)[0]
    return {"review": review.text, "sentiment": prediction}

@app.post("/predict/hugging-face")
async def predict(review: Review):
    prediction = pipeline(review.text)[0]
    return {
        "sentiment": prediction["label"],
        "review": float(prediction["score"])
    }
