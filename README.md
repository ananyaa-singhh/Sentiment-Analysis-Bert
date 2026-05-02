# Sentiment Analysis using DistilBERT
A sentiment analysis web app that classifies movie reviews as positive or negative using DistilBERT fine-tuned on the IMDB dataset.

## Live Demo
[Live Demo](https://sentiment-analysis-bert-xw3xebn8ycsv4xpnmhb8vg.streamlit.app/)

## Project Overview
- Fine-tuned DistilBERT on 2000 IMDB movie reviews
- Achieved **84.2% accuracy** on test data
- Built interactive UI using Streamlit
- Supports both single and batch review analysis

## Tech Stack
- Python
- HuggingFace Transformers
- DistilBERT
- Streamlit
- PyTorch

## Model Performance
 Metric | Score 
 Accuracy | 84.2% 
 Dataset | IMDB (2000 samples) 
 Model | DistilBERT-base-uncased 

## How to Run
1. Clone the repository
2. Install dependencies
3. Run the app

## Project Structure
sentiment-analysis-bert/
├── sentiment_app.py
├── requirements.txt
└── README.md
