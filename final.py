print("🚀 Script started...")

import os
import time
from semi_final import retrieve_documents, format_citations
from upsert import get_pinecone_index, upsert_embeddings_to_pinecone
from pre_process import fetch_news, save_news_to_json
from full_pre_process import run_preprocessing_pipeline
from summarize import summarize_and_store_in_batches, load_bart_model_and_tokenizer, load_spacy_model
from embeddings import generate_news_embeddings, load_sentence_bert_model
print("🚀 Script started 1...")

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def autonomous_source_selection(query, pinecone_api_key, pinecone_index, retrieval_model, bart_tokenizer, bart_model, nlp, top_k=3):

    """
    Retrieves relevant articles from Pinecone.
    If insufficient results are found, fetches fresh articles, processes them, upserts them to Pinecone, and re-runs retrieval.
    """
    print("\n🔄 Retrieving articles from Pinecone...")
    results = retrieve_documents(query, pinecone_index, retrieval_model, top_k=top_k)
    print(f"📊 Retrieved {len(results)} articles from Pinecone before fetching new ones.")

    if len(results) < top_k:
        print("🔍 Insufficient results. Fetching fresh news articles...")

        page_size = 1 if len(results) == 0 else 3
        new_articles = fetch_news(api_key=NEWS_API_KEY, queries=[query], page_size=page_size)


        if not new_articles:
            print("⚠️ No fresh news articles found. Returning existing results.")
            return results

        print("\n📰 **Articles Before Upserting:**")
        for article in new_articles:
            print(f"📌 {article.get('title', 'No Title')}")

        save_news_to_json(new_articles, filename="news_with_topics.json")

        preprocessed_df = run_preprocessing_pipeline(json_filename="news_with_topics.json", 
                                                     output_csv="preprocessed_news_with_topics.csv")

        if preprocessed_df is None or preprocessed_df.empty:
            print("❌ Preprocessing failed. Returning existing results.")
            return results

        summarize_and_store_in_batches(preprocessed_df, bart_tokenizer, bart_model, nlp, batch_size=3, output_file="summarized_news_batch.csv")


        generate_news_embeddings(input_csv="summarized_news_batch.csv", 
                                 output_csv="summarized_news_with_embeddings_batch.csv")

        upsert_embeddings_to_pinecone("summarized_news_with_embeddings_batch.csv", pinecone_api_key, "brand-news-2")

        # ✅ **Exponential Backoff for Pinecone Indexing**
        delay = 5
        for _ in range(3):  
            time.sleep(delay)
            print(f"⏳ Waiting {delay} seconds for Pinecone indexing...")
            delay *= 2  

        print("\n🔄 Re-attempting retrieval from Pinecone...")
        results = retrieve_documents(query, pinecone_index, retrieval_model, top_k=top_k)
        print(f"📊 Retrieved {len(results)} articles after upserting.")

    else:
        print("✅ Sufficient results found in Pinecone.")

    print("\n--- Retrieved Articles ---")
    if results:
        for doc in results:
            title = doc['metadata'].get('title', 'No Title')
            summary = doc['metadata'].get('summary', 'No Summary')
            url = doc['metadata'].get('url', '')
            print(f"📌 {title}\n📝 {summary}\n🔗 {url}\n")
    else:
        print("⚠️ No articles retrieved.")

    return results

# ✅ **Main Execution**
if __name__ == "__main__":
    print("🚀 Script started..2.")

    # ✅ Load Models
    bart_tokenizer, bart_model, bart_timestamp = load_bart_model_and_tokenizer()
    print("🚀 Script started.3..")

    nlp, spacy_timestamp = load_spacy_model()  # ✅ Corrected loading
    print("🚀 Script started..4.")

    retrieval_model, device = load_sentence_bert_model()
    print("🚀 Script started..5.")


    # ✅ Initialize Pinecone
    pinecone_index = get_pinecone_index(PINECONE_API_KEY, index_name="brand-news-2")
    print("🚀 Script started..6.")


    if pinecone_index is None:
        print("❌ Pinecone index initialization failed. Exiting.")
        exit(1)

    user_query = input("Enter your query: ")
    retrieved_docs = autonomous_source_selection(user_query, PINECONE_API_KEY, pinecone_index, retrieval_model, bart_tokenizer, bart_model, nlp, top_k=3)


    print("\n--- Retrieved Articles ---")
    for doc in retrieved_docs:
        title = doc['metadata'].get('title', 'No Title')
        summary = doc['metadata'].get('summary', 'No Summary')
        url = doc['metadata'].get('url', '')
        print(f"Title: {title}\nSummary: {summary}\nURL: {url}\n")

    citations = format_citations(retrieved_docs)
    print("\n--- Citations ---")
    print(citations)
