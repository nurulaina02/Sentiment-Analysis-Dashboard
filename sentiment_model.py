from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]
