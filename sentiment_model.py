import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def run_sentiment(input_path, output_path):
    df = pd.read_csv(input_path)
    df["sentiment"] = df["clean_text"].apply(get_sentiment)
    df.to_csv(output_path, index=False)
    print("Sentiment analysis completed!")

if __name__ == "__main__":
    run_sentiment("data/processed/clean_reviews.csv",
                  "data/processed/sentiment_results.csv")
