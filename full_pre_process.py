import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# ----------------------------
# Function: Fetch Random Articles
# ----------------------------
def fetch_random_articles(api_key, query="news", language="en", page_size=24):
    """
    Fetches a set of random articles from NewsAPI using a generic query.

    :param api_key: Your NewsAPI key.
    :param query: Generic search query (default: "news").
    :param language: Language filter (default: 'en').
    :param page_size: Number of articles to fetch (default: 24).
    :return: List of news articles.
    """
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["status"] != "ok":
            print("Error fetching articles:", data.get("message", "Unknown error"))
            return []
        articles = data.get("articles", [])
        return articles
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []

# ----------------------------
# Function: Fetch Full Article Text
# ----------------------------
def fetch_full_article(url):
    """
    Fetches the full article text from a given URL using BeautifulSoup.
    Extracts the main content from paragraph (<p>) tags.
    Handles redirects and skips consent pages.

    :param url: The URL of the article.
    :return: The full article text, or None if fetching fails.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if response.url.startswith('https://consent.yahoo.com') or response.status_code == 401:
            print(f"⚠️ Skipping consent page: {url}")
            return None
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([para.get_text() for para in paragraphs])
        return full_text.strip()
    except Exception as e:
        print(f"⚠️ Error fetching article from {url}: {e}")
        return None

# ----------------------------
# Function: Truncate Text
# ----------------------------
def truncate_text(text, max_tokens=1024):
    """
    Truncates the text to fit within a specified token limit.
    Approximate ratio: 1024 tokens ~ 780 words.

    :param text: The original text.
    :param max_tokens: The maximum number of tokens (words) allowed.
    :return: The truncated text.
    """
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])
    return text

# ----------------------------
# Function: Preprocess Articles
# ----------------------------


def preprocess_articles(df):
    """
    Preprocesses news articles from a DataFrame:
      - Fetches the full article if the content is missing or too short.
      - Cleans the text by removing unnecessary HTML and truncating it.
      - Adds a processing timestamp.

    :param df: DataFrame containing raw article data.
    :return: A DataFrame with processed articles (without topic information).
    """
    processed_data = []
    for _, row in df.iterrows():
        title = row["title"]
        source = row["source"]
        url = row["url"]
        content = row["content"]

        # If content is missing or too short, try to fetch the full article
        if not content or len(content.split()) < 50:
            content = fetch_full_article(url)
        if not content:
            continue
        content = truncate_text(content)

        processed_data.append({
            "title": title,
            "source": source,
            "url": url,
            "cleaned_text": content,
            "timestamp": int(time.time())  # Processing timestamp (UNIX format)
        })

    return pd.DataFrame(processed_data)

# ----------------------------
# Function: Save News to JSON
# ----------------------------
def save_news_to_json(news_list, filename="news_with_topics.json"):
    """
    Saves the list of news articles to a JSON file.

    :param news_list: List of news articles.
    :param filename: Filename for the JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news_list, f, ensure_ascii=False, indent=4)

# ----------------------------
# Function: Load News from JSON
# ----------------------------
def load_news_from_json(filename="news_with_topics.json"):
    """
    Loads news articles from a JSON file and converts them to a DataFrame.

    :param filename: Path to the JSON file.
    :return: A Pandas DataFrame containing the news articles.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            print("Data is not in the expected format.")
            return None
    except Exception as e:
        print(f"Error loading JSON file {filename}: {e}")
        return None

# ----------------------------
# Function: Run Preprocessing Pipeline (Using JSON Data)
# ----------------------------
def run_preprocessing_pipeline(json_filename="news_with_topics.json", output_csv="preprocessed_news_with_topics.csv"):
    """
    Executes the complete preprocessing pipeline:
      1. Loads raw news articles from a JSON file.
      2. Converts the data into a DataFrame.
      3. Applies preprocessing to clean and truncate the text.
      4. Saves the processed data to a CSV file.

    :param json_filename: JSON file containing raw news articles.
    :param output_csv: Filename for the output CSV with preprocessed data.
    :return: A DataFrame with preprocessed articles.
    """
    df = load_news_from_json(json_filename)
    if df is None:
        return None
    preprocessed_df = preprocess_articles(df)
    preprocessed_df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessing complete. Data saved as {output_csv}")
    return preprocessed_df

# ----------------------------
# Function: Run Full Random News Pipeline
# ----------------------------
def run_random_news_pipeline(api_key=NEWS_API_KEY, json_filename="news_with_topics.json", output_csv="preprocessed_news_with_topics.csv"):
    """
    Executes a full pipeline to fetch random news articles, save them to a JSON file,
    and then preprocess and save them to a CSV file.

    :param api_key: Your NewsAPI key.
    :param json_filename: Filename to save raw news articles.
    :param output_csv: Filename for the preprocessed CSV file.
    :return: A DataFrame with preprocessed articles.
    """
    # Fetch 24 random articles using a generic "news" query
    articles = fetch_random_articles(api_key, query="news", language="en", page_size=24)
    # Save the fetched articles to JSON
    save_news_to_json(articles, filename=json_filename)
    # Run the preprocessing pipeline on the saved JSON data
    preprocessed_df = run_preprocessing_pipeline(json_filename=json_filename, output_csv=output_csv)
    return preprocessed_df

# ----------------------------
# Example Execution (for testing purposes)
# ----------------------------
# if __name__ == "__main__":
#     df_processed = run_random_news_pipeline()
#     if df_processed is not None:
#         print("Sample Processed Articles DataFrame:")
#         print(df_processed.head())
