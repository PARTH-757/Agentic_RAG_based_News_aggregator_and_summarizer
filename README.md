# Agentic RAG-based News Retriever and Summarizer

## Introduction  
With the rapid growth of digital news, efficiently retrieving and summarizing relevant content has become essential. This project combines powerful Transformer based Models (LLMs) such as BART and BERT with an innovative Agentic Retrieval-Augmented Generation (RAG) framework to deliver a robust news retrieval and summarization system. It fetches news articles, generates concise summaries, and stores them in the Pinecone vector database. It improves comprehension using Agentic techniques where the model checks if news articles are already present in the RAG; if not, it autonomously fetches, summarizes, and stores them in Pinecone before retrieving for the user.


---

## Abstract  
This project presents an Agentic RAG-based approach that integrates retrieval-based search with context-aware summarization. Utilizing NewsAPI for real-time news fetching, BERT-based embeddings for semantic document representation, and BART for abstractive summarization, the system efficiently retrieves, processes, and summarizes news articles. Designed for researchers, journalists, and general users, it offers concise insights with high accuracy and efficiency on limited hardware.

---

## Architecture & Methodology  

### System Components  
1. **News Retrieval Module:**  
   Fetches up-to-date news articles using the NewsAPI.

2. **Preprocessing Unit:**  
   Cleans and tokenizes the retrieved news articles to prepare them for embedding.

3. **Agentic Retrieval-Augmented Generation (RAG) Pipeline:**  
   - **Agentic Component:** An autonomous agent that dynamically refines search queries and re-ranks articles based on relevance and recency to adapt retrieval to user needs.  
   - **Embedding Layer:** Uses Sentence-BERT (SBERT) to create high-quality semantic embeddings of news content.  
   - **Retrieval Mechanism:** Performs similarity searches with vector embeddings and cosine similarity to fetch the most contextually relevant articles, if the similarity function fails to retrieve any article the agent atomatically retrieves a set of articles, summarizes them and stores them as vectors in Pinecone vector database, before returning the same to the user.

4. **Summarization Module:**  
   Utilizes BART for abstractive summarization, generating human-like summaries while preserving essential information.

5. **Output Layer:**  
   Displays summaries in a clean, structured format with options to filter results, and provides links to original articles with citations.

6. **User Interface:**  
   Built with Streamlit, offering a lightweight, interactive UI optimized for ease of use.


---

## Implementation Details  

- **Backend:** Python  
- **News Source:** NewsAPI for fetching real-time news  
- **Vector Database:** Pinecone used to store and query vector embeddings efficiently  
- **Similarity Search:** Cosine similarity on BERT-based embeddings for precise article retrieval  
- **Summarization:** Pretrained BART fine-tuned for news article summarization  
- **User Interface:** Streamlit for a responsive, lightweight frontend  
- **Deployment:** Optimized to run on CPU with minimal hardware resources  

---

