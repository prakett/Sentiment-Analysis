import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "saved_roberta"

print("Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully!")

labels = {
    0: "Negative",
    1: "Positive"
}


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "prediction": labels[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


from model import predict_sentiment

predict_sentiment("I absolutely loved this movie.")