document.addEventListener("DOMContentLoaded", () => {
    const totalCharacterPresent = document.getElementById('total-character-present');
    const analyzeButtonSentiment = document.getElementById('analyze-sentiment-button-click');
    const inputFeedbackText = document.getElementById("input-feedback-text");
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
            sentimentResult.textContent = `${result.prediction} (${result.confidence}% confidence)`;
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