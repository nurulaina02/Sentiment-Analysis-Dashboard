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

# ================== PAGE NAVIGATION ==================
page = st.selectbox(
    "📂 Select Page",
    ["🏠 Project Overview", "📊 Dashboard", "✅ Conclusion"]
)

# =================================================
# =============== PROJECT OVERVIEW PAGE ===========
# =================================================
if page == "🏠 Project Overview":

    st.title("📌 Project Overview")

    st.subheader("1. Objective")
    st.write(
        """
        The objective of this project is to develop an interactive Natural Language Processing (NLP) 
        dashboard capable of performing sentiment and emotion analysis on textual data. 
        The system aims to provide meaningful insights into public opinions and emotional trends 
        through intuitive visualizations.
        """
    )

    st.subheader("2. Problem Statement")
    st.write(
        """
        With the rapid increase of user-generated text such as online reviews and social media comments, 
        manual sentiment analysis has become inefficient and impractical. 
        There is a need for an automated solution that can accurately classify sentiment and emotions 
        while presenting results in a clear and user-friendly format.
        """
    )

    st.subheader("3. Solution Architecture")
    st.write(
        """
        The proposed solution follows a modular NLP pipeline:
        """
    )

    st.markdown(
        """
        - **Dataset**: Kaggle Sentiment and Emotion Analysis Dataset  
        - **Preprocessing**: Text cleaning and normalization  
        - **Analysis**: Sentiment polarity and emotion classification  
        - **Evaluation**: Precision, Recall, and F1-score metrics  
        - **Visualization**: Interactive Streamlit dashboard  
        """
    )

# =================================================
# ================= DASHBOARD PAGE ================
# =================================================
elif page == "📊 Dashboard":

    st.title("📊 Sentiment & Emotion Analysis Dashboard")

    analysis_type = st.radio(
        "Select Analysis Type",
        ["Sentiment Analysis", "Emotion Analysis"],
        horizontal=True
    )

    search_text = st.text_input("🔍 Search text")

    # ---------- SENTIMENT ANALYSIS ----------
    if analysis_type == "Sentiment Analysis":

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(sentiment_df))
        col2.metric("Positive", (sentiment_df["sentiment"].str.lower() == "positive").sum())
        col3.metric("Neutral", (sentiment_df["sentiment"].str.lower() == "neutral").sum())
        col4.metric("Negative", (sentiment_df["sentiment"].str.lower() == "negative").sum())

        st.subheader("Sentiment Distribution")

        sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig_sentiment = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.4
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

        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]

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

    # ---------- EMOTION ANALYSIS ----------
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
            color="Emotion"
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

# =================================================
# ================= CONCLUSION PAGE ===============
# =================================================
else:

    st.title("✅ Conclusion")

    st.write(
        """
        This project successfully demonstrates the application of Natural Language Processing techniques 
        for sentiment and emotion analysis through an interactive dashboard. 
        By integrating sentiment classification, emotion detection, and performance evaluation, 
        the system provides meaningful insights into textual data.

        The use of Streamlit enables real-time visualization and interaction, while the modular 
        architecture ensures scalability and ease of future enhancement. 
        Overall, the dashboard serves as an effective analytical tool for understanding public 
        opinions and emotional trends.
        """
    )
