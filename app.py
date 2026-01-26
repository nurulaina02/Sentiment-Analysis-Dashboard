import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd

from preprocessing import clean_text
from model import predict_sentiment

# Load dataset
df = pd.read_csv("data/Sentiment Analysis Dataset.csv")

# Assume columns: text, sentiment
df["clean_text"] = df["text"].apply(clean_text)
df["predicted"] = df["clean_text"].apply(lambda x: predict_sentiment(x)[0])

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),

    dcc.Textarea(
        id="input-text",
        placeholder="Enter text to analyze...",
        style={"width": "100%", "height": 120}
    ),

    html.Button("Analyze", id="analyze-btn", n_clicks=0),

    html.Div(id="output-text", style={"marginTop": 20}),

    dcc.Graph(
        figure=px.histogram(
            df, x="predicted",
            title="Overall Sentiment Distribution"
        )
    )
])

@app.callback(
    Output("output-text", "children"),
    Input("analyze-btn", "n_clicks"),
    Input("input-text", "value")
)
def analyze_text(n_clicks, text):
    if not text:
        return ""
    label, score = predict_sentiment(clean_text(text))
    return f"Predicted Sentiment: {label} (Confidence: {score:.2f})"

if __name__ == "__main__":
    app.run_server(debug=True)
