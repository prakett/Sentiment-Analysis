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


THRESHOLD = 0.70  # Change this value as needed


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

    negative_probability = probabilities[0][0].item()
    positive_probability = probabilities[0][1].item()

    if positive_probability >= THRESHOLD:
        prediction = "Positive"
        confidence = positive_probability
    else:
        prediction = "Negative"
        confidence = negative_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "positive": round(positive_probability * 100, 2),
            "negative": round(negative_probability * 100, 2)
        },
        "logits": {
            "positive": round(outputs.logits[0][1].item(), 4),
            "negative": round(outputs.logits[0][0].item(), 4)
        },
        "threshold": THRESHOLD
    }


from model import predict_sentiment

predict_sentiment("I absolutely loved this movie.")