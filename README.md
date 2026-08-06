# 🚀 Multilingual Sentiment Analysis using Fine-Tuned RoBERTa

A multilingual sentiment analysis web application built using **RoBERTa**, **FastAPI**, and **Flask**.

The application automatically detects the input language, translates non-English text into English using **MarianMT**, and predicts sentiment using a fine-tuned **RoBERTa** model.

---

## ✨ Features

- Fine-tuned RoBERTa for sentiment analysis
- FastAPI REST API
- Flask web interface
- Automatic language detection
- MarianMT translation
- Supports English, Hindi, German, French, Spanish, and Italian
- Confidence score prediction
- CUDA GPU acceleration
- Modular architecture
- Easy to extend with additional languages

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | Flask |
| Deep Learning | PyTorch |
| NLP Library | Hugging Face Transformers |
| Translation | MarianMT |
| Language Detection | langdetect |
| Model | RoBERTa-base |
| GPU Support | CUDA |

---

# 📂 Project Structure

```text
Sentiment-Analysis/
│
├── saved_roberta/
│
├── static/
│
├── templates/
│
├── api.py
├── app.py
├── model.py
├── translator.py
├── requirements.txt
├── README.md
│
└── ...
```

---

# 🚀 Quick Start

## 1. Fork the Repository (Recommended)

Click the **Fork** button on GitHub to create your own copy.

Clone your fork:

```bash
git clone https://github.com/prakett/Sentiment-Analysis.git
cd Sentiment-Analysis
```

or clone the original repository:

```bash
git clone https://github.com/prakett/Sentiment-Analysis.git
cd Sentiment-Analysis
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies inside virtual environment

```bash
pip install -r requirements.txt
```

---

## 4. Run the FastAPI Backend

```bash
python -m uvicorn api:app --reload
```

The backend will start on:

```
http://127.0.0.1:8000
```

Interactive API Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 5. Run the Flask Frontend

Open another terminal.

Activate the virtual environment again.

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 🌍 Supported Languages

The application automatically detects the input language and translates non-English text into English before sentiment analysis.

Current supported languages:

- 🇺🇸 English
- 🇮🇳 Hindi
- 🇩🇪 German
- 🇫🇷 French
- 🇪🇸 Spanish
- 🇮🇹 Italian

Additional languages can easily be added by extending the `MODEL_NAMES` dictionary in `translator.py`.

---

# ⚙️ System Workflow

```text
                 User
                   │
                   ▼
          Flask Web Interface
                   │
                   ▼
            FastAPI Backend
                   │
                   ▼
        Language Detection
                   │
        ┌──────────┴──────────┐
        │                     │
     English          Other Language
        │                     │
        │          MarianMT Translation
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
        Fine-Tuned RoBERTa Model
                   │
                   ▼
     Sentiment + Confidence Score
                   │
                   ▼
            JSON Response
```

---

# 📊 API Example

## Request

```json
{
    "text": "Dieser Film war fantastisch."
}
```

## Response

```json
{
    "detected_language": "de",
    "translated_text": "This movie was fantastic.",
    "prediction": "Positive",
    "confidence": 98.74
}
```

---

# 🧠 Model Information

- Model: Fine-Tuned RoBERTa-base
- Task: Binary Sentiment Classification
- Translation: MarianMT
- Language Detection: langdetect
- Framework: Hugging Face Transformers
- Inference: PyTorch

---

# 🤝 Contributing

Contributions are welcome!

### 1. Fork the repository

Click the **Fork** button on GitHub.

### 2. Clone your fork

```bash
git clone https://github.com/<your-github-username>/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 3. Create a new branch

```bash
git checkout -b feature/my-feature
```

### 4. Make your changes

Implement your feature or fix the issue.

### 5. Commit your changes

```bash
git add .
git commit -m "Describe your changes"
```

### 6. Push your branch

```bash
git push origin feature/my-feature
```

### 7. Open a Pull Request

Create a Pull Request describing your changes.

---

# 🔄 Updating Your Fork

Add the original repository as an upstream remote:

```bash
git remote add upstream https://github.com/prakett/Sentiment-Analysis.git
```

Fetch the latest changes:

```bash
git fetch upstream
```

Merge the latest updates:

```bash
git checkout main
git merge upstream/main
```

---

# 📦 Updating Dependencies

Whenever new packages are added:

```bash
pip freeze > requirements.txt
```

Commit the updated `requirements.txt`.

---

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. It helps others discover the project and supports future development.
