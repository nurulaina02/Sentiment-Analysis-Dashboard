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

# ================== SIDEBAR NAVIGATION ==================
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Select Page",
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
        dashboard that performs sentiment and emotion analysis on textual data. The system aims to 
        provide meaningful insights into public opinions and emotional patterns through clear and 
        intuitive visualizations.
        """
    )

    st.subheader("2. Problem Statement")
    st.write(
        """
        The rapid growth of user-generated textual data such as online reviews and social media comments 
        makes manual analysis inefficient and impractical. An automated solution is required to 
        accurately classify sentiment and emotions while presenting results in a user-friendly interface.
        """
    )

    st.subheader("3. Solution Architecture")
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

        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(sentiment_df))
        col2.metric("Positive", (sentiment_df["sentiment"].str.lower() == "positive").sum())
        col3.metric("Neutral", (sentiment_df["sentiment"].str.lower() == "neutral").sum())
        col4.metric("Negative", (sentiment_df["sentiment"].str.lower() == "negative").sum())

        # Sentiment Distribution Pie Chart
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

        # -------- FIXED CLASSIFICATION REPORT --------
        st.subheader("Sentiment Classification Performance")

        if "predicted_sentiment" not in sentiment_df.columns:
            sentiment_df["predicted_sentiment"] = sentiment_df["sentiment"]

        report = classification_report(
            sentiment_df["sentiment"],
            sentiment_df["predicted_sentiment"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        # SAFE label selection (prevents KeyError)
        expected_labels = ["Positive", "Neutral", "Negative", "macro avg", "weighted avg"]
        available_labels = [lbl for lbl in expected_labels if lbl in report_df.index]
        report_df = report_df.loc[available_labels]

        # Rename index nicely
        report_df.index = [
            lbl.title() if "avg" not in lbl else lbl.replace(" avg", " Avg").title()
            for lbl in report_df.index
        ]

        # Select and rename columns
        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]

        st.dataframe(report_df.round(3), use_container_width=True)

        # Data Explorer
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

# =================================================
# ================= CONCLUSION PAGE ===============
# =================================================
else:

    st.title("✅ Conclusion")

    st.write(
        """
        This project successfully demonstrates the application of Natural Language Processing 
        techniques for sentiment and emotion analysis through an interactive dashboard. 
        By integrating sentiment classification, emotion detection, and performance evaluation, 
        the system provides meaningful insights into textual data.

        The use of Streamlit enables real-time visualization and interaction, while the modular 
        architecture supports scalability and future enhancements such as real-time data integration 
        and advanced machine learning models.
        """
    )
