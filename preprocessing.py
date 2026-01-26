import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text
