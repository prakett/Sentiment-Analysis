import pickle

with open("sentiment_model2.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

model = sentiment_model["model"]
vectorizer = sentiment_model["vectorizer"]
label_map = sentiment_model["label_map"]  

print("=" * 50)
print("Sentiment Analysis")
print("Type 'exit' to quit.")
print("=" * 50)

while True:
    text = input("\nEnter a sentence: ").strip()

    if text.lower() == "exit":
        print("Goodbye!")
        break

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    print(f"\nPredicted Sentiment: {label_map[prediction]}")
    print(f"Confidence: {probabilities[prediction] * 100:.2f}%")

    print("\nProbabilities:")
    print(f"Negative : {probabilities[0] * 100:.2f}%")
    print(f"Positive : {probabilities[1] * 100:.2f}%")