import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

df = pd.read_csv("dataset\\sentiment.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

text_columns = ['cleaned_text']

for col in text_columns:
    df[col] = df[col].apply(preprocess)

df.to_csv("preprocessed_dataset.csv", index=False)

print(df.head())

