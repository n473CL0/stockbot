# models/sentiment_llm.py
from transformers import pipeline
import streamlit as st

# Load FinBERT once
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

finbert = load_finbert()

def analyze_sentiment_finbert(titles, threshold=0.2):
    """
    Run FinBERT sentiment analysis on a list of titles.
    Returns only numerical sentiment scores in range [-1, 1],
    where + means positive, - means negative, 0 means neutral/weak.
    """
    if not titles:
        return []

    results = finbert(titles, truncation=True)
    scores = []
    for r in results:
        label = r["label"].upper()
        prob = r["score"]

        if label == "POSITIVE":
            val = +prob
        elif label == "NEGATIVE":
            val = -prob
        else:
            val = 0.0

        # Zero out weak signals below threshold
        scores.append(val if abs(val) >= threshold else 0.0)

    return scores
