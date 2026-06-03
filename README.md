# Sentiment Analysis using DistilBERT
A sentiment analysis web app that classifies movie reviews as positive or negative using DistilBERT fine-tuned on the IMDB dataset.

## Live Demo
[Live Demo](https://sentiment-analysis-bert-xw3xebn8ycsv4xpnmhb8vg.streamlit.app/)

## Project Overview
- Fine-tuned DistilBERT on 8000 IMDB movie reviews
- Achieved **92 % accuracy** on test data
- Built interactive UI using Streamlit
- Supports both single and batch review analysis

## Tech Stack
- Python
- HuggingFace Transformers
- DistilBERT
- Streamlit
- Plotly
- PyTorch

## Model Performance
 Metric | Score 
 Accuracy | ~92%+ 
 Dataset | IMDB (8,000 samples) 
 Batch size | 16 |
 Model | DistilBERT-base-uncased 
 Regularisation | Weight decay + Warmup |

## How to Run
1. Clone the repository
2. Install dependencies
3. Run the app

## Project Structure
sentiment-analysis-bert/
├── sentiment_app.py           
├── sentiment_ananlysis.ipynb  
├── requirements.txt          
├── my_sentiment_model/        
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model.safetensors
└── README.md

