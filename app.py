import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import clean_text
from sentiment import predict_sentiment

# ---------------------------------------------
# Page configuration
# ---------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment Analysis Dashboard")
st.write("Sentiment analysis using dataset hosted on GitHub.")

# ---------------------------------------------
# Load dataset from GitHub
# ---------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/refs/heads/main/sentiment_clean.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset:\n{e}")
        return None

df = load_data()

if df is None:
    st.stop()

if "text" not in df.columns:
    st.error("The dataset does not contain a column named 'text'.")
    st.stop()

# ---------------------------------------------
# Run sentiment analysis
# ---------------------------------------------
@st.cache_data
def analyze_sentiment(dataframe):
    sentiments = []
    confidences = []

    for txt in dataframe["text"]:
        cleaned_txt = clean_text(txt)
        label, score = predict_sentiment(cleaned_txt)
        sentiments.append(label)
        confidences.append(score)

    df_new = dataframe.copy()
    df_new["sentiment"] = sentiments
    df_new["confidence"] = confidences
    return df_new

with st.spinner("Running sentiment analysis..."):
    df = analyze_sentiment(df)

# ---------------------------------------------
# Sidebar options
# ---------------------------------------------
st.sidebar.header("Dashboard Options")
show_preview = st.sidebar.checkbox("Show dataset preview")

# ---------------------------------------------
# Visualizations
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig_dist = px.histogram(
        df,
        x="sentiment",
        color="sentiment",
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    fig_box = px.box(
        df,
        x="sentiment",
        y="confidence",
        title="Confidence Score by Sentiment"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------
# User Input Sentiment Analysis
# ---------------------------------------------
st.subheader("🔍 Analyze Your Own Text")

user_text = st.text_area("Type text here...")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        lbl, sc = predict_sentiment(clean_text(user_text))
        st.success(f"Sentiment: {lbl}")
        st.info(f"Confidence: {sc:.2f}")

# ---------------------------------------------
# Show dataset preview
# ---------------------------------------------
if show_preview:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))
