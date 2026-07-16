# 🩺 Medical AI Chatbot

An AI-powered medical chatbot built using **Python, PyTorch, Transformer Model, Retrieval-Based Question Answering, and Streamlit**. The chatbot provides medical information by answering user questions using a trained custom transformer model and a medical knowledge dataset.

> **Note:** This chatbot is designed for educational and informational purposes only. It is **not a replacement for professional medical advice, diagnosis, or treatment.**

---

# Features

- 🧠 Custom Transformer-based Medical AI
- 📚 Medical Knowledge Retrieval
- 🌍 Supports Multiple Languages
- 🔍 Automatic Language Detection
- 💬 AI-powered Medical Question Answering
- 🎨 Modern Streamlit Web Interface
- ⚡ Fast Response Generation
- 📱 Responsive UI
- ☁️ Easy Deployment on Streamlit Cloud or Railway

---

# Project Structure

```
Medical-AI/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── medical_chatbot_dataset.csv
│   ├── medquad.csv
│   └── pubmedqa.parquet
│
├── models/
│   ├── medical_transformer.pth
│   └── vocab.json
│
├── assets/
│   ├── background.png
│   └── girl.png
│
└── src/
    ├── chatbot.py
    ├── retrieval.py
    ├── tokenizer.py
    ├── transformer_model.py
    ├── preprocess.py
    ├── train.py
    ├── dataset.py
    ├── inference.py
    ├── evaluate.py
    └── config.py
```

---

# Technologies Used

- Python
- PyTorch
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- LangDetect
- Deep Translator

---

# Dataset

This project uses publicly available medical datasets:

- MedQuAD
- PubMedQA

The datasets are preprocessed and combined into a single training dataset.

---

# Model

The chatbot uses:

- Custom Medical Tokenizer
- Custom Transformer Neural Network
- Retrieval-Based Response System

The trained model is stored in:

```
models/medical_transformer.pth
```

Vocabulary:

```
models/vocab.json
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Medical-AI.git

cd Medical-AI
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# Training

To train the transformer model:

```bash
python src/train.py
```

---

# Evaluation

```bash
python src/evaluate.py
```

---

# Example Questions

- What are the symptoms of diabetes?
- How is hypertension treated?
- What causes fever?
- What is asthma?
- How can I reduce blood pressure?
- What are the symptoms of pneumonia?
- What is migraine?
- How is malaria diagnosed?

---

# Screenshots

## Home Page

Add screenshot here

```
assets/home.png
```

---

## Chat Interface

Add screenshot here

```
assets/chat.png
```

---

# Deployment

The application can be deployed using:

- Streamlit Community Cloud
- Railway
- Render
- Docker
- VPS

---

# Requirements

Example:

```
streamlit
torch
pandas
numpy
scikit-learn
nltk
langdetect
deep-translator
```

---

# Future Improvements

- Medical image analysis
- Appointment booking
- User authentication
- Medical report summarization
- Voice assistant
- Hospital locator
- Drug interaction checker
- Electronic Health Record (EHR) integration

---

# Disclaimer

This project is intended for educational and research purposes only.

The chatbot may generate incorrect or incomplete information. Always consult a qualified healthcare professional for medical advice.

---

# Author

**Your Name**

Final Year Project

Medical AI Chatbot

---

# License

This project is licensed under the MIT License.

---

## Star the Repository

If you found this project useful, please consider giving it a ⭐ on GitHub.

Afshan-Anshad
