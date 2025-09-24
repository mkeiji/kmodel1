import os
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ---------------------------
# Utility functions
# ---------------------------

def print_evaluation_title(title: str):
    print("\n" + "=" * 50)
    print(f"📊 {title}")
    print("=" * 50)

def plot_confusion_matrix(y_true, y_pred, labels, filename=None):
    # Folder to save confusion matrices (created if it doesn't exist)
    CONF_MATRIX_DIR = os.path.join(os.path.dirname(__file__), "confusion-matrix")
    os.makedirs(CONF_MATRIX_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if filename:
        full_path = os.path.join(CONF_MATRIX_DIR, filename)  # save inside folder
        plt.savefig(full_path)
        print(f"Confusion matrix saved as {full_path}")
    # close to not show image
    plt.close()

def plot_metrics(y_true, y_pred, labels, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Confusion matrix saved to file
    plot_confusion_matrix(y_true, y_pred, labels, filename=f"{model_name}_confusion.png")
    print(f"Confusion matrix saved as {model_name}_confusion.png")

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("data/IMDB Dataset.csv").head(2000)  # smaller subset for speed
X_test = df["review"].tolist()
y_test = df["sentiment"].tolist()

# ---------------------------
# Load Naive Bayes pipeline
# ---------------------------
nb_pipeline = joblib.load("models/naive_bayes/sentiment_pipeline.pkl")

# ---------------------------
# Create Hugging Face pipeline
# ---------------------------
model_path = "models/hugging_face"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
hugging_face_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt")

# ---------------------------
# Naive Bayes evaluation
# ---------------------------
print_evaluation_title("Naive Bayes Evaluation")
y_pred_nb = nb_pipeline.predict(X_test)
plot_metrics(y_test, y_pred_nb, labels=["negative", "positive"], model_name="naive_bayes")

# ---------------------------
# Hugging Face evaluation
# ---------------------------
print_evaluation_title("Hugging Face Evaluation")
y_pred_hf = [hugging_face_pipeline(text, truncation=True, max_length=128)[0]['label'] for text in X_test]

# Map labels if Hugging Face returns "LABEL_0"/"LABEL_1"
y_pred_hf = ["negative" if p == "LABEL_0" else "positive" for p in y_pred_hf]
plot_metrics(y_test, y_pred_hf, labels=["negative", "positive"], model_name="hugging_face")
