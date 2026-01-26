import pandas as pd
from utils.preprocessing import clean_text
from models.sentiment_model import predict_sentiment

def analyze_dataset(csv_path):
    df = pd.read_csv(csv_path)

    sentiments = []
    scores = []

    for text in df["text"]:
        cleaned = clean_text(text)
        label, score = predict_sentiment(cleaned)
        sentiments.append(label)
        scores.append(score)

    df["sentiment"] = sentiments
    df["confidence"] = scores
    return df
