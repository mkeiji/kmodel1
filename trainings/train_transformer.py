import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.transformers
from mlflow.models.signature import infer_signature

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("data/IMDB Dataset.csv").head(5000)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ---------------------------
# 2. Tokenizer
# ---------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["review"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# 3. Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ---------------------------
# 4. Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./models/hugging_face/training_args",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
)

# ---------------------------
# 5. Metrics for evaluation
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------------
# 6. MLflow logging
# ---------------------------
mlflow.set_experiment("Sentiment Analysis - DistilBERT")

with mlflow.start_run():
    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # Log hyperparameters
    mlflow.log_params({
        "model_name": model_name,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "num_epochs": training_args.num_train_epochs
    })

    # Create a signature from a small batch of test inputs
    sample_inputs = tokenizer(list(test_df["review"].head(5)), padding=True, truncation=True, max_length=128, return_tensors="pt")
    signature = infer_signature(sample_inputs, trainer.predict(test_dataset).predictions.argmax(axis=-1))

    # Log model with signature and pip requirements (no input_example)
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        signature=signature,
        pip_requirements=[
            f"torch=={torch.__version__}",
            f"transformers=={transformers.__version__}",
            "datasets>=2.14.0",
            "accelerate>=0.20.0",
            "scikit-learn>=1.2.0",
        ],
    )

print("✅ Training complete! Metrics & model logged to MLflow.")
