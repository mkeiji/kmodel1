from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import joblib

class Review(BaseModel):
    text: str

# ---------- Naive Bayes pipeline ----------
nb_pipeline = joblib.load("models/naive_bayes/sentiment_pipeline.pkl")

# ---------- Hugging Face model ----------
model_path = "models/hugging_face"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
hf_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt")

# ---------- FastAPI app ----------
app = FastAPI(title="Sentiment Analysis - Kmodel")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Endpoints ----------
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/predict/naive-bayes")
def predict_naive_bayes(review: Review):
    prediction = nb_pipeline.predict([review.text])[0]
    return {"review": review.text, "sentiment": prediction}

@app.post("/predict/hugging-face")
def predict_hugging_face(review: Review):
    prediction = hf_pipeline(review.text, truncation=True, max_length=128)[0]
    return {"review": review.text, "sentiment": prediction["label"], "score": float(prediction["score"])}
