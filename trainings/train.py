import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Ensure folders exist
def ensure_model_dirs(relative_path):
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    full_path = os.path.abspath(os.path.join(base_dir, relative_path))
    os.makedirs(full_path, exist_ok=True)
    return full_path

# Paths
DATA_PATH = "data/IMDB Dataset.csv"
MODEL_PATH = os.path.join(ensure_model_dirs("models/naive_bayes"), "sentiment_pipeline.pkl")

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download it first.")

    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df["review"]
    y = df["sentiment"]

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("⚙️ Training pipeline...")
    nb_pipeline = make_pipeline(
        CountVectorizer(stop_words="english"),
        MultinomialNB()
    )
    nb_pipeline.fit(X_train, y_train)

    print("🔎 Evaluating...")
    y_pred = nb_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"✅ Pipeline trained with accuracy: {acc:.2f}")

    print("💾 Saving pipeline locally...")
    joblib.dump(nb_pipeline, MODEL_PATH)

    # -----------------------
    # MLflow logging
    # -----------------------
    mlflow.set_experiment("Sentiment Analysis - Naive Bayes")

    # Create signature only (no input example)
    signature = infer_signature(X_test, y_pred)

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({
            "vectorizer": "CountVectorizer(stop_words=english)",
            "classifier": "MultinomialNB",
            "test_size": 0.2,
        })

        # log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        # log the pipeline with signature only
        mlflow.sklearn.log_model(
            sk_model=nb_pipeline,
            artifact_path="model",
            signature=signature
        )

    print("🎉 Done! Pipeline and metrics logged to MLflow with signature (no warnings).")

if __name__ == "__main__":
    train()
