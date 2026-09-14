from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


# Homepage
@app.get("/")
def home():
    return FileResponse("src/index.html")


# Sentiment prediction API
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
        "confidence": sentiment["confidence"],
        "probabilities": sentiment["probabilities"]
    }


# Serve everything inside src/
# This must come AFTER the API routes above.
app.mount(
    "/",
    StaticFiles(directory="src", html=True),
    name="frontend"
)