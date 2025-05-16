import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer

def load_sentence_bert_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads the Sentence-BERT model and returns it along with the device used.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name).to(device)
    return model, device

def generate_embedding(text, model):
    """
    Converts text into a dense numerical vector using Sentence-BERT.
    """
    embedding = model.encode(text, convert_to_tensor=True).to("cpu").numpy()
    return embedding.tolist()

def generate_news_embeddings(input_csv="summarized_news_batch.csv", output_csv="summarized_news_with_embeddings_batch.csv"):
    """
    Loads summarized news, generates embeddings, and saves the results.
    """
    # Load summarized news data
    df = pd.read_csv(input_csv, on_bad_lines="warn")
    print(df.head())  # Check if all columns are correct

    # Load Sentence-BERT model
    model, device = load_sentence_bert_model()

    # Generate embeddings and per-row timestamps
    df["embedding"] = df["summary"].fillna("").apply(lambda x: generate_embedding(x, model) if x.strip() else None)
    df["timestamp"] = df["summary"].apply(lambda _: int(time.time()))  # ✅ Individual timestamp per article

    # Save embeddings along with other metadata (excluding "topic") to CSV
    df[["title", "source", "url", "summary", "keywords", "embedding", "timestamp"]].to_csv(output_csv, index=False)

    print(f"✅ Embedding generation complete! {len(df)} articles saved to '{output_csv}'.")

# # Example usage
# if __name__ == "__main__":
#     generate_news_embeddings()
