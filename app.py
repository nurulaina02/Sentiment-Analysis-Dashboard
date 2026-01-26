import streamlit as st
import pandas as pd
import plotly.express as px
import importlib.util
import os

# ---------- LOAD sentiment_model.py ----------
model_path = os.path.join("model", "sentiment_model.py")
spec = importlib.util.spec_from_file_location("sentiment_model", model_path)
sentiment_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentiment_model)

predict_sentiment = sentiment_model.predict_sentiment

# ---------- LOAD preprocessing.py ----------
utils_path = os.path.join("utils", "preprocessing.py")
spec_utils = importlib.util.spec_from_file_location("preprocessing", utils_path)
preprocessing = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(preprocessing)

clean_text = preprocessing.clean_text

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/sentiment_clean.csv")

df = load_data()
df["clean_text"] = df["text"].astype(str).apply(clean_text)

@st.cache_resource
def run_sentiment(texts):
    return texts.apply(predict_sentiment)

sentiments = run_sentiment(df["clean_text"])
df["sentiment"] = sentiments.apply(lambda x: x[0])
df["confidence"] = sentiments.apply(lambda x: x[1])

st.subheader("Dataset Preview")
st.dataframe(df.head())

sentiment_count = df["sentiment"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

fig = px.pie(
    sentiment_count,
    names="Sentiment",
    values="Count",
    title="Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Live Sentiment Prediction")

user_text = st.text_area("Enter text to analyze sentiment")

if st.button("Analyze Sentiment"):
    if user_text.strip():
        label, score = predict_sentiment(clean_text(user_text))
        st.success(f"Sentiment: {label}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.warning("Please enter some text.")
