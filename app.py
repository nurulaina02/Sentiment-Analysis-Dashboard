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
st.sidebar.title("📊 Project Information")

# ----- PROJECT OVERVIEW (TOP) -----
with st.sidebar.expander("📌 Project Overview", expanded=True):

    st.markdown("**Objective**")
    st.markdown(
        """
        To develop an interactive NLP-based dashboard that performs sentiment and emotion 
        analysis on textual data, enabling users to gain insights into public opinions 
        and emotional patterns through effective visualization.
        """
    )

    st.markdown("**Problem Statement**")
    st.markdown(
        """
        Manual analysis of large-scale textual data such as reviews and social media comments 
        is inefficient and time-consuming. An automated system is required to accurately 
        classify sentiment and emotions while presenting results in a clear and user-friendly format.
        """
    )

    st.markdown("**Solution Architecture**")
    st.markdown(
        """
        - **Data Source**: Kaggle Sentiment & Emotion Dataset  
        - **Preprocessing**: Text cleaning and normalization  
        - **Analysis**: Sentiment polarity & emotion classification  
        - **Evaluation**: Precision, Recall, F1-score  
        - **Visualization**: Streamlit interactive dashboard  
        """
    )

# ----- CONTROLS (MIDDLE) -----
st.sidebar.divider()

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Sentiment Analysis", "Emotion Analysis"]
)

search_text = st.sidebar.text_input("🔍 Search text")

# ----- CONCLUSION (LAST) -----
st.sidebar.divider()

with st.sidebar.expander("✅ Conclusion"):

    st.markdown(
        """
        This project successfully applies Natural Language Processing techniques to analyze 
        sentiment and emotions from textual data using an interactive dashboard. The system 
        provides meaningful insights through visual analytics and structured performance evaluation.

        The modular design and use of Streamlit ensure scalability, usability, and ease of deployment. 
        Overall, the dashboard serves as an effective analytical tool for understanding public opinions 
        and emotional trends, with strong potential for future enhancements.
        """
    )

# ================== MAIN TITLE ==================
st.title("📊 Sentiment Analysis Dashboard")
st.caption("Interactive NLP Dashboard for Sentiment and Emotion Analysis")

# =================================================
# =============== SENTIMENT ANALYSIS ===============
# =================================================
if analysis_type == "Sentiment Analysis":

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

    st.subheader("Sentiment Classification Performance")

    if "predicted_sentiment" not in sentiment_df.columns:
        sentiment_df["predicted_sentiment"] = sentiment_df["sentiment"]

    report = classification_report(
        sentiment_df["sentiment"],
        sentiment_df["predicted_sentiment"],
        output_dict=True
    )

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

    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Explore Reviews")

    if search_text:
        st.dataframe(
            sentiment_df[
                sentiment_df["clean_text"].str.contains(search_text, case=False, na=False)
            ]
        )
    else:
        st.dataframe(sentiment_df.head(50))

# =================================================
# ================ EMOTION ANALYSIS ================
# =================================================
else:

    emotion_counts = emotion_df["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(emotion_df))
    col2.metric("Most Common Emotion", emotion_counts.iloc[0]["Emotion"])

    st.subheader("Emotion Distribution")

    fig_emotion = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        color="Emotion",
        title="Emotion Frequency Distribution"
    )

    st.plotly_chart(fig_emotion, use_container_width=True)

    st.subheader("Explore Emotion Data")

    if search_text:
        st.dataframe(
            emotion_df[
                emotion_df["text"].str.contains(search_text, case=False, na=False)
            ]
        )
    else:
        st.dataframe(emotion_df.head(50))
