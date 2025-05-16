import torch
from sentence_transformers import SentenceTransformer
from upsert import  get_pinecone_index
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()





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
    # Generate embedding for the query using Sentence-BERT
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()

    # Perform the similarity search in Pinecone
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k * 2,  # Fetch extra results for better ranking
        include_metadata=True
    )

    # Filter results based on similarity score
    filtered_results = [doc for doc in results['matches'] if doc.get('score', 0) >= score_threshold]

    # Sort purely by similarity score (highest first)
    sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)

    return sorted_results[:top_k]  # Return only the top-k results

def format_citations(documents):
    """
    Formats a list of documents into a Markdown-formatted citation string.
    """
    citations = []
    for doc in documents:
        title = doc['metadata'].get('title', 'No Title')
        url = doc['metadata'].get('url', '')
        citations.append(f"[{title}]({url})")
    return "\n".join(citations)

def run_retrieval_pipeline(query, index, model, top_k=3):
    """
    Runs the full retrieval pipeline purely based on similarity ranking.
    """
    documents = retrieve_documents(query, index, model, top_k=top_k)
    citations = format_citations(documents)
    return documents, citations
# Example usage:
# if __name__ == "__main__":
#     # Load the Sentence-BERT model
#     model, device = load_sentence_bert_model()

#     # Example: assume that the Pinecone index is already initialized and assigned to the variable `index`
#     #index = get_pinecone_index(api_key, index_name)

#     pinecone_api_key = os.getenv("PINECONE_API_KEY")  # ✅ Correct

#     index = get_pinecone_index(pinecone_api_key)

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
