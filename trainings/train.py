import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training pipeline...")
    nb_pipeline = make_pipeline(
        CountVectorizer(stop_words="english"),
        MultinomialNB()
    )

    nb_pipeline.fit(X_train, y_train)

    acc = nb_pipeline.score(X_test, y_test)
    print(f"✅ Pipeline trained with accuracy: {acc:.2f}")

    print("💾 Saving pipeline...")
    joblib.dump(nb_pipeline, MODEL_PATH)

    print("🎉 Done! Pipeline saved.")

if __name__ == "__main__":
    train()
