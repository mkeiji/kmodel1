# Sentiment Analysis - Kmodel

This project is a FastAPI service with a simple frontend that uses a machine learning model trained on the IMDB Movie Reviews Dataset to classify reviews as positive or negative.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
Place the dataset IMDB Dataset.csv in the root folder.
```

2. Train the model:

```bash
make train
```
This will generate sentiment_model.pkl and vectorizer.pkl.

3. Running Locally
```bash
make run
```
Open http://localhost:8000 in your browser. You can use the simple HTML frontend to input reviews and see sentiment predictions.

## API Endpoints
GET / - Serves the HTML frontend.
POST /predict - Accepts a review form field and returns JSON:

```json
{
    "review": "This movie was amazing!",
    "sentiment": "positive"
}
```

## Docker
Build and run the container:

```bash
make docker-build
make docker-run
```
The app will be accessible at http://localhost:8000.
