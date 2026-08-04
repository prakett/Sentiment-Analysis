import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "saved_roberta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

labels = {0: "Negative", 1: "Positive"}

print("Sentiment Analyzer")
print("Type 'exit' to quit.\n")

while True:
    text = input("You: ")

    if text.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    prediction = torch.argmax(probs).item()

