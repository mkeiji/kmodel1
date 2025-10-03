# Sentiment Analysis - Kmodel

This is a FastAPI service with a simple frontend that uses a machine learning model trained on the IMDB Movie Reviews Dataset to classify reviews as positive or negative.

My main goal with this project is to learn the basics and workflow of machine learning, including data preprocessing, model training, evaluation, and deployment.

# Learning Roadmap (So Far)

- **Naive Baseline Models**
  - Learned what a baseline is and why it’s important.  
  - *Project*: Built a simple baseline (e.g., majority class predictor).  
- **Classical ML Models (Logistic Regression, SVM, etc.)**
  - Learned how to train/evaluate classical models with scikit-learn.  
  - *Project*: Implemented `evaluate.py` and compared against baselines.  
- **Transformers for NLP (DistilBERT)**
  - Learned how Hugging Face’s `Trainer` works (training, epochs, evaluation).  
  - *Project*: Fine-tuned DistilBERT on IMDB dataset.  
- **Experiment Comparison (Classical vs Transformers)**
  - Learned how advanced models compare to classical ones.  
  - *Project*: Used `evaluate.py` to compare transformer vs. traditional ML.  
    - _note_: Found transformers outperform classical models, but the results are misleading due to 
      small dataset size and lack of hyperparameter tuning.
- **Experiment Tracking with MLflow**
  - Learned about auto-logging of metrics, params, artifacts.  
  - *Project*: Wrapped DistilBERT training with MLflow tracking.  

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
Place the dataset IMDB Dataset.csv in the root folder.
```

2. Train the naive-bayes model:

```bash
make train-1
```
This will generate sentiment_model.pkl and vectorizer.pkl.

3. Train the hugging-face model:

```bash
make train-2
```
This will generate the model inside the `hugging_face` folder.

4. Run MLflow to track experiments (optional):

```bash
make mlflow
```
This will start an MLflow server with models trained logged.

5. Running Locally
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
