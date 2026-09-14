import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"

TRAIN_FILE = "dataset4\\train.csv"
VAL_FILE = "dataset4\\validation.csv"
TEST_FILE = "dataset4\\test.csv"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

#Embedding BASEDMODEL
#Preprocessing .,#, stop words
#Lametization


train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

for df in (train_df, val_df, test_df):
    df["cleaned_text"] = df["cleaned_text"].fillna("")
    df["target"] = df["target"].astype(int)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["cleaned_text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("target", "labels")
val_dataset = val_dataset.rename_column("target", "labels")
test_dataset = test_dataset.rename_column("target", "labels")

train_dataset = train_dataset.remove_columns(
    [c for c in ["cleaned_text", "__index_level_0__"] if c in train_dataset.column_names]
)
val_dataset = val_dataset.remove_columns(
    [c for c in ["cleaned_text", "__index_level_0__"] if c in val_dataset.column_names]
)
test_dataset = test_dataset.remove_columns(
    [c for c in ["cleaned_text", "__index_level_0__"] if c in test_dataset.column_names]
)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels,
    )["accuracy"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir="roberta_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1
        )
    ],
)

trainer.train()

print("\nValidation Results")
print(trainer.evaluate(val_dataset))

print("\nTest Results")
print(trainer.evaluate(test_dataset))

trainer.save_model("saved_roberta")
tokenizer.save_pretrained("saved_roberta")

print("\nDone!")