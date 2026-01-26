from flask import Flask, render_template
import pandas as pd
import plotly.express as px

from model.sentiment_model import predict_sentiment
from utils.preprocessing import clean_text

app = Flask(__name__)

@app.route("/")
def index():
    # Load dataset
    df = pd.read_csv("data/sentiment_clean.csv")

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Predict sentiment
    sentiments = df["clean_text"].apply(predict_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])

    # Count sentiment
    sentiment_count = df["sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]

    # Plotly pie chart
    fig = px.pie(
        sentiment_count,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution"
    )

    graph_html = fig.to_html(full_html=False)

    return render_template("index.html", graph=graph_html)

if __name__ == "__main__":
    app.run(debug=True)
