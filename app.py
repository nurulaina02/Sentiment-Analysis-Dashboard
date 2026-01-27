import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_sentiment_data.csv"
    )
    emotion_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/Sentiment-Analysis-Dashboard/main/combined_emotion_small.csv"
    )
    return sentiment_df, emotion_df


sentiment_df, emotion_df = load_data()

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Dashboard Controls")

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Sentiment Analysis", "Emotion Analysis"]
)

search_text = st.sidebar.text_input("🔍 Search text")

# ================== TITLE ==================
st.title("📊 Sentiment Analysis Dashboard")
st.caption("Interactive NLP Dashboard for Sentiment and Emotion Analysis")

# =================================================
# =============== SENTIMENT ANALYSIS ===============
# =================================================
if analysis_type == "Sentiment Analysis":

    # ----- KPI METRICS -----
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(sentiment_df))
    col2.metric(
        "Positive",
        (sentiment_df["sentiment"].str.lower() == "positive").sum()
    )
    col3.metric(
        "Neutral",
        (sentiment_df["sentiment"].str.lower() == "neutral").sum()
    )
    col4.metric(
        "Negative",
        (sentiment_df["sentiment"].str.lower() == "negative").sum()
    )

    # ----- SENTIMENT PIE CHART -----
    st.subheader("Sentiment Distribution")

    sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.4,
        title="Overall Sentiment Breakdown"
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

    # ----- CLASSIFICATION REPORT TABLE (EXPLICIT FORMAT) -----
    st.subheader("Sentiment Classification Performance")

    # Ensure predicted labels exist
    if "predicted_sentiment" not in sentiment_df.columns:
        sentiment_df["predicted_sentiment"] = sentiment_df["sentiment"]

    y_true = sentiment_df["sentiment"]
    y_pred = sentiment_df["predicted_sentiment"]

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    # Build clean table explicitly
    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.loc[
        ["Positive", "Neutral", "Negative", "macro avg", "weighted avg"]
    ]

    report_df.index = [
        "Positive",
        "Neutral",
        "Negative",
        "Macro Avg",
        "Weighted Avg"
    ]

    report_df = report_df[
        ["precision", "recall", "f1-score", "support"]
    ]

    report_df.columns = [
        "Precision",
        "Recall",
        "F1-Score",
        "Support"
    ]

    report_df = report_df.round(3)

    # Display table
    st.dataframe(report_df, use_container_width=True)

    # ----- DATA EXPLORER -----
    st.subheader("Explore Reviews")

    if search_text:
        filtered_df = sentiment_df[
            sentiment_df["clean_text"].str.contains(
                search_text, case=False, na=False
            )
        ]
        st.dataframe(filtered_df)
    else:
        st.dataframe(sentiment_df.head(50))


# =================================================
# ================ EMOTION ANALYSIS ================
# =================================================
else:

    # ----- EMOTION KPIs -----
    emotion_counts = emotion_df["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(emotion_df))
    col2.metric("Most Common Emotion", emotion_counts.iloc[0]["Emotion"])

    # ----- EMOTION BAR CHART (MULTI-COLOUR) -----
    st.subheader("Emotion Distribution")

    fig_emotion = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        color="Emotion",
        title="Emotion Frequency Distribution"
    )

    st.plotly_chart(fig_emotion, use_container_width=True)

    # ----- DATA EXPLORER -----
    st.subheader("Explore Emotion Data")

    if search_text:
        filtered_df = emotion_df[
            emotion_df["text"].str.contains(
                search_text, case=False, na=False
            )
        ]
        st.dataframe(filtered_df)
    else:
        st.dataframe(emotion_df.head(50))
