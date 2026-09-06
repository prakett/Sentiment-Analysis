document.addEventListener("DOMContentLoaded", () => {
    const totalCharacterPresent = document.getElementById('total-character-present');
    const analyzeButtonSentiment = document.getElementById('analyze-sentiment-button-click');
    const inputFeedbackText = document.getElementById("input-feedback-text");

    analyzeButtonSentiment.addEventListener('click', () => {
        const text = inputFeedbackText.value.trim();
        
    })
    // to keep track of each key pressed.
    inputFeedbackText.addEventListener('keyup', () => {
        const text = inputFeedbackText.value.trim();
        totalCharacterPresent.textContent = `${text.length} / 500 characters`;
    })

})