from transformers import pipeline
import pandas as pd

emotion_classifier = pipeline("text-classification",
                              model="j-hartmann/emotion-english-distilroberta-base",
                              return_all_scores=False)

def detect_emotion(text):
    return emotion_classifier(text)[0]["label"]

def run_emotion(input_path, output_path):
    df = pd.read_csv(input_path)
    df["emotion"] = df["clean_text"].apply(detect_emotion)
    df.to_csv(output_path, index=False)
