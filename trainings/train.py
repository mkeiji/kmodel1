import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
MODEL_PATH = os.path.join(ensure_model_dirs("models/naive_bayes"), "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(ensure_model_dirs("models/naive_bayes"), "vectorizer.pkl")

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download it first.")

    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df["review"]
    y = df["sentiment"]

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training model...")
    vectorizer = CountVectorizer(stop_words="english")
    model = MultinomialNB()

    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)

    acc = model.score(vectorizer.transform(X_test), y_test)
    print(f"✅ Model trained with accuracy: {acc:.2f}")

    print("💾 Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("🎉 Done! Model and vectorizer saved.")

if __name__ == "__main__":
    train()
