document.addEventListener("DOMContentLoaded", () => {
    const totalCharacterPresent = document.getElementById('total-character-present');
    const analyzeButtonSentiment = document.getElementById('analyze-sentiment-button-click');
    const inputFeedbackText = document.getElementById("input-feedback-text");
    const positiveResultDisplay = document.getElementById("Positive-result");
    const neutralResultDisplay = document.getElementById("Neutral-result");
    const negativeResultDisplay = document.getElementById("Negative-result");

    const sentimentResult = document.getElementById("sentiment-result");

    analyzeButtonSentiment.addEventListener('click', async () => {
        const text = inputFeedbackText.value.trim();

        if (!text) {
            sentimentResult.textContent = "Please enter feedback first.";
            return;
        }

        analyzeButtonSentiment.disabled = true;
        sentimentResult.textContent = "Analyzing...";

        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
            }

            const result = await response.json();

            if (result.prediction === "Positive") {
                positiveResultDisplay.textContent = `${result.confidence.toFixed(1)}%`;
                neutralResultDisplay.textContent = `${(100 - result.confidence).toFixed(1)}%`;

                negativeResultDisplay.textContent = `${(100 - result.confidence - 10).toFixed(1)}%`
            }
            else if (result.prediction === "Negative") {
                negativeResultDisplay.textContent = `${result.confidence.toFixed(1)}%`;
                neutralResultDisplay.textContent = `${(100 - result.confidence).toFixed(1)}%`;

                positiveResultDisplay.textContent = `${(100 - result.confidence - 10).toFixed(1)}%`
            }
            

        } catch (error) {
            sentimentResult.textContent = "Could not connect to the sentiment API.";
            console.error(error);
        } finally {
            analyzeButtonSentiment.disabled = false;
        }
    });

    // to keep track of each key pressed.
    inputFeedbackText.addEventListener('keyup', () => {
        const text = inputFeedbackText.value.trim();
        totalCharacterPresent.textContent = `${text.length} / 500 characters`;
    })

})