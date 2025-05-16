import streamlit as st
import time
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from semi_final import retrieve_documents, format_citations
from upsert import get_pinecone_index, upsert_embeddings_to_pinecone
from pre_process import fetch_news, save_news_to_json
from full_pre_process import run_preprocessing_pipeline
from summarize import summarize_and_store_in_batches, load_spacy_model
from embeddings import generate_news_embeddings, load_sentence_bert_model

# ✅ API Keys
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Load BART Model and Tokenizer (Optimized for CPU)
@st.cache_resource
def load_bart_model_and_tokenizer(model_name="facebook/bart-large-cnn"):
    device = "cpu"  # ✅ Explicitly set CPU usage
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    with torch.no_grad():  # ✅ Reduce memory usage
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    
    return tokenizer, model, int(time.time())  # ✅ Return load timestamp

# ✅ Load All Required Models and Index
@st.cache_resource
def load_models():
    bart_tokenizer, bart_model, bart_timestamp = load_bart_model_and_tokenizer()
    nlp, spacy_timestamp = load_spacy_model()
    retrieval_model, _ = load_sentence_bert_model()
    pinecone_index = get_pinecone_index(PINECONE_API_KEY, "brand-news-2")
    return bart_tokenizer, bart_model, bart_timestamp, nlp, retrieval_model, pinecone_index

# ✅ Initialize Models & Pinecone
bart_tokenizer, bart_model, bart_timestamp, nlp, retrieval_model, pinecone_index = load_models()

# ✅ Autonomous Source Selection Logic
def autonomous_source_selection(query, top_k=3):
    st.write("🔄 Retrieving articles from Pinecone...")
    results = retrieve_documents(query, pinecone_index, retrieval_model, top_k=top_k)
    st.write(f"📊 Retrieved {len(results)} articles before fetching new ones.")

    if len(results) < top_k:
        st.write("🔍 Fetching fresh news articles...")
        page_size = 1 if len(results) == 0 else 3
        new_articles = fetch_news(api_key=NEWS_API_KEY, queries=[query], page_size=page_size)

        if not new_articles:
            st.write("⚠️ No fresh news found. Returning existing results.")
            return results

        save_news_to_json(new_articles, filename="news_with_topics.json")

        preprocessed_df = run_preprocessing_pipeline(
            json_filename="news_with_topics.json",
            output_csv="preprocessed_news_with_topics.csv"
        )

        if preprocessed_df is None or preprocessed_df.empty:
            st.write("❌ Preprocessing failed. Returning existing results.")
            return results

        summarize_and_store_in_batches(preprocessed_df, bart_tokenizer, bart_model, nlp, batch_size=3, output_file="summarized_news_batch.csv")

        generate_news_embeddings(input_csv="summarized_news_batch.csv", output_csv="summarized_news_with_embeddings_batch.csv")

        upsert_embeddings_to_pinecone("summarized_news_with_embeddings_batch.csv", PINECONE_API_KEY, "brand-news-2")

        # ✅ Exponential Backoff for Indexing
        delay = 5
        for _ in range(3):
            time.sleep(delay)
            st.write(f"⏳ Waiting {delay} seconds for Pinecone indexing...")
            delay *= 2  

        st.write("🔄 Re-attempting retrieval...")
        results = retrieve_documents(query, pinecone_index, retrieval_model, top_k=top_k)

    return results

# ✅ Streamlit UI
st.title("📰 News Retrieval & Summarization")

user_query = st.text_input("Enter your query:")

if st.button("Search"):
    if not user_query:
        st.warning("⚠️ Please enter a search query.")
    else:
        st.write("🔄 Processing...")
        retrieved_docs = autonomous_source_selection(user_query, top_k=3)

        if retrieved_docs:
            for doc in retrieved_docs:
                title = doc['metadata'].get('title', 'No Title')
                summary = doc['metadata'].get('summary', 'No Summary')
                url = doc['metadata'].get('url', '#')
                st.subheader(title)
                st.write(summary)
                st.markdown(f"[🔗 Read More]({url})")
        else:
            st.warning("⚠️ No articles retrieved.")

        citations = format_citations(retrieved_docs)
        st.markdown("### 📑 Citations")
        st.code(citations, language="text")
