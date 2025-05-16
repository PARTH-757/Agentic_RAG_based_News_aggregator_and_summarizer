import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# Optionally, load environment variables from a .env file (only for development)
# pip install python-dotenv
# from dotenv import load_dotenv
# load_dotenv()  # This loads variables from a .env file if present

# Global API key for NewsAPI, loaded from environment variable
# NEWS_API_KEY = "84c58d17e4f541178e6d60c426ec3fac"
if not NEWS_API_KEY:
    raise ValueError("Please set the NEWS_API_KEY environment variable.")

def fetch_news(api_key, queries, language="en", page_size=24):
    """
    Fetches news articles from NewsAPI based on multiple queries.
    """
    all_articles = []
    
    # Define topic-based keywords for assignment
    topic_keywords = {
        'Sports': ['sports'],
        'Politics': ['politics'],
        'Technology': ['technology', 'tech'],
        'Climate': ['climate', 'environment'],
        'General': []  # For articles that don't match any keyword
    }
    
    for query in queries:
        url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data["status"] != "ok":
                print(f"Error fetching articles for '{query}':", data.get("message", "Unknown error"))
                continue
            articles = data.get("articles", [])
            for article in articles:
                title = article.get("title", "").lower()
                content = article.get("content", "").lower() if article.get("content") else ""
                assigned_topic = 'General'
                for topic, keywords in topic_keywords.items():
                    if any(keyword in title or keyword in content for keyword in keywords):
                        assigned_topic = topic
                        break
                article['topic'] = assigned_topic
                # Convert publishedAt to a datetime object if possible
                if 'publishedAt' in article:
                    try:
                        article['publishedAt'] = datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00"))
                    except Exception as e:
                        pass
                all_articles.append(article)
        except requests.exceptions.RequestException as e:
            print(f"Request failed for '{query}':", e)
    
    return all_articles

def save_news_to_json(news_list, filename="news_with_topics.json"):
    """Save news articles to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news_list, f, ensure_ascii=False, indent=4, default=str)

def news_to_dataframe(news_list):
    """Convert list of news articles to a Pandas DataFrame."""
    return pd.DataFrame([{
        "title": article.get("title", ""),
        "content": article.get("content", ""),
        "source": article.get("source", {}).get("name", ""),
        "url": article.get("url", ""),
        "publishedAt": article.get("publishedAt", ""),
        "topic": article.get("topic", "General")
    } for article in news_list])

def run_random_news_pipeline(api_key=NEWS_API_KEY, queries=["technology", "sports", "politics", "climate"], output_json="news_with_topics.json", output_csv="preprocessed_news_with_topics.csv"):
    """Executes a full pipeline to fetch news, save to JSON, and preprocess to CSV."""
    articles = fetch_news(api_key, queries=queries, language="en", page_size=24)
    
    # Add a timestamp to each raw article
    for article in articles:
        article["timestamp"] = int(time.time())
    
    save_news_to_json(articles, filename=output_json)
    
    # Load JSON back as a DataFrame and preprocess if needed
    df = pd.read_json(output_json)
    df_preprocessed = df  # Here you could call additional preprocessing if required
    df_preprocessed.to_csv(output_csv, index=False)
    
    return df_preprocessed

# Example usage:
if __name__ == "__main__":
    # Run the pipeline with desired queries
    queries = ["technology", "artificial intelligence", "climate", "politics"]
    preprocessed_df = run_random_news_pipeline(api_key=NEWS_API_KEY, queries=queries)
    print(preprocessed_df.head())
