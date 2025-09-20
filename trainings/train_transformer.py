import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split

# 1. Load a subset of IMDB (first 5000 reviews for speed)
df = pd.read_csv("data/IMDB Dataset.csv").head(5000)

# Map "positive"/"negative" → 1/0
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["review"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Ensure folders exist
def ensure_model_dirs(relative_path):
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    full_path = os.path.abspath(os.path.join(base_dir, relative_path))
    os.makedirs(full_path, exist_ok=True)
    return full_path

# 3. Model
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 4. Training args (backward-compatible)
training_args_path = ensure_model_dirs("models/hugging_face/training_args")
training_args = TrainingArguments(
    output_dir=training_args_path,
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

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 6. Train
trainer.train()

# 7. Save
model_dir = ensure_model_dirs("models/hugging_face")
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print("✅ Training complete! Model saved to models/hugging_face")
