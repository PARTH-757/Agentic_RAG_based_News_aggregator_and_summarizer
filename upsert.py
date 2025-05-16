import pandas as pd
import numpy as np
import pinecone
import time
import torch
from sentence_transformers import SentenceTransformer


def get_pinecone_index(api_key, index_name="brand-news-2"):
    """
    Initializes the Pinecone client with the given API key and returns the specified index.

    :param api_key: Your Pinecone API key.
    :param index_name: The name of the index to connect to (default: "brand-news-2").
    :return: The Pinecone index instance.
    """
    pc = pinecone.Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def upsert_embeddings_to_pinecone(csv_file, pinecone_api_key, index_name):
    """
    Loads summarized news data with embeddings from a CSV file, converts the embeddings from strings
    to numpy arrays, prepares metadata with timestamps, and upserts the vectors into a Pinecone index.

    :param csv_file: Path to the CSV file containing summarized news with embeddings.
    :param pinecone_api_key: Your Pinecone API key.
    :param index_name: Name of the Pinecone index to upsert into.
    :return: The upsert response from Pinecone.
    """
    df = pd.read_csv(csv_file)

    # Convert embeddings from string to NumPy array
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    embeddings = np.stack(df['embedding'].values)

    # Get current timestamp
    current_timestamp = int(time.time())

    # Prepare metadata with timestamp included
    metadata = df[['title', 'source', 'url', 'summary', 'keywords']].to_dict(orient='records')
    for meta in metadata:
        meta['timestamp'] = current_timestamp

    # Create unique IDs
    ids = [f"news_{i}" for i in range(len(df))]

    # Prepare vectors for upserting
    vectors = list(zip(ids, embeddings.tolist(), metadata))

    # Initialize Pinecone and connect to index
    index = get_pinecone_index(pinecone_api_key, index_name)

    # Upsert into Pinecone
    upsert_response = index.upsert(vectors=vectors)

    print(f"✅ Successfully upserted {len(vectors)} embeddings into Pinecone.")
    return upsert_response


def load_sentence_bert_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads the Sentence-BERT model and returns it along with the device used.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name).to(device)
    return model, device


def retrieve_documents(query, index, model, top_k=3, score_threshold=0.18):
    """
    Retrieves the top-k relevant documents from the Pinecone index for a given query,
    purely based on semantic similarity.
    """
    # Generate embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().tolist()

    # Perform similarity search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k * 2,
        include_metadata=True
    )

    # Filter results based on similarity score
    filtered_results = [doc for doc in results['matches'] if doc.get('score', 0) >= score_threshold]

    # Sort by similarity score (highest first)
    sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)

    return sorted_results[:top_k]  # Return only the top-k results


def format_citations(documents):
    """
    Formats a list of documents into a Markdown-formatted citation string.
    """
    return "\n".join([f"[{doc['metadata'].get('title', 'No Title')}]({doc['metadata'].get('url', '')})" for doc in documents])


def run_retrieval_pipeline(query, index, model, top_k=3):
    """
    Runs the full retrieval pipeline.
    """
    documents = retrieve_documents(query, index, model, top_k=top_k)
    citations = format_citations(documents)
    return documents, citations


# if __name__ == "__main__":
#     # Load Sentence-BERT model
#     model, device = load_sentence_bert_model()

#     # Initialize Pinecone index
#     PINECONE_API_KEY = " pcsk_5GP7j7_F3H3tzJN77rt8wrFJ5EGLETQ7TU82rLPL4ZbPQRh6uSoCn19TinpTZitDgm8Ttf"  # Replace with actual API key
#     INDEX_NAME = "brand-news-2"
#     index = get_pinecone_index(PINECONE_API_KEY, INDEX_NAME)

#     # Query input
#     query = input("Enter your query: ")
#     documents, citations = run_retrieval_pipeline(query, index, model, top_k=3)

#     print("\n--- Retrieved Articles ---")
#     for doc in documents:
#         title = doc['metadata'].get('title', 'No Title')
#         summary = doc['metadata'].get('summary', 'No Summary')
#         url = doc['metadata'].get('url', '')
#         print(f"Title: {title}")
#         print(f"Summary: {summary}")
#         print(f"URL: {url}\n")

#     print("\n--- Citations ---")
#     print(citations)
