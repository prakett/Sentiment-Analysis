from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import predict_sentiment
from translator import translate_to_english

app = FastAPI(
    title="RoBERTa Sentiment Analysis API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "RoBERTa Sentiment Analysis API is running!"
    }


@app.post("/predict")
def predict(data: TextInput):

    translation = translate_to_english(data.text)

    sentiment = predict_sentiment(
        translation["translated_text"]
    )

    return {
        "detected_language": translation["language"],
        "translated_text": translation["translated_text"],
        "prediction": sentiment["prediction"],
        "confidence": sentiment["confidence"]
    }