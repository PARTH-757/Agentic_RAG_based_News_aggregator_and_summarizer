import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

def load_preprocessed_news(csv_filename="preprocessed_news_with_topics.csv"):
    """
    Loads preprocessed news articles from a CSV file into a Pandas DataFrame.
    Ensures that the timestamp column is properly converted to a datetime format.

    :param csv_filename: The path to the CSV file.
    :return: A Pandas DataFrame containing the preprocessed news articles.
    """
    df = pd.read_csv(csv_filename)

    # ✅ Convert timestamp column to datetime if it exists
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df

# Usage:
# df = load_preprocessed_news()
# print(df.head())
