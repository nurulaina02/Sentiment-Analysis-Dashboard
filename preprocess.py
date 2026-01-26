import re
import pandas as pd

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(output_path, index=False)
    print("Preprocessing completed!")

if __name__ == "__main__":
    preprocess_csv("data/raw/reviews.csv", "data/processed/clean_reviews.csv")
