import torch
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_models = {}

MODEL_NAMES = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}


def load_model(language):

    if language in loaded_models:
        return loaded_models[language]

    model_name = MODEL_NAMES[language]

    print(f"Loading translator: {language} -> English")

    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model = MarianMTModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    loaded_models[language] = (tokenizer, model)

    return tokenizer, model


def translate_to_english(text):

    try:
        language = detect(text)

    except Exception:
        language = "en"

    if language == "en":

        return {
            "language": "English",
            "translated_text": text
        }

    if language not in MODEL_NAMES:

        return {
            "language": language,
            "translated_text": text
        }

    tokenizer, model = load_model(language)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        translated = model.generate(**inputs)

    english = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return {
        "language": language,
        "translated_text": english
    }