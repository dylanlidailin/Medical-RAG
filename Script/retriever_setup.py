# retriever_setup.py
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# --- Configuration ---
# Using a fine-tuned PubMedBERT model via sentence-transformers for ease of use
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings" 
EMBEDDING_DIM = 768

# --- Global Components (Loaded Once) ---
try:
    # SentenceTransformer simplifies the process of getting mean-pooled sentence embeddings
    EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Loaded PubMedBERT Embedder: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"Error loading embedder: {e}")
    EMBEDDER = None
    
# --- Data and Vector Index ---
DOCUMENTS = []
VECTOR_INDEX = None # Will store the Faiss index

def load_and_index_documents(file_path: str = "knowledge_base.txt"):
    """
    Loads text, chunks it (simple line-by-line chunking), embeds it,
    and builds a Faiss index for fast search.
    """
    global DOCUMENTS, VECTOR_INDEX

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Simple chunking: each line is a document/chunk
            DOCUMENTS = [line.strip() for line in f if line.strip()]
        
        if not DOCUMENTS or EMBEDDER is None:
            return False

        # 1. Generate Embeddings
        print(f"Generating embeddings for {len(DOCUMENTS)} chunks...")
        embeddings = EMBEDDER.encode(DOCUMENTS, convert_to_numpy=True, show_progress_bar=True)
        
        # 2. Build Faiss Index
        # We use IndexFlatL2 for simple Euclidean distance (or equivalent dot product for normalized vectors)
        # L2 distance is fast and effective for semantic search.
        VECTOR_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
        VECTOR_INDEX.add(embeddings) # Add the vectors to the index
        
        print(f"Faiss index built with {VECTOR_INDEX.ntotal} vectors.")
        return True

    except Exception as e:
        print(f"Error during indexing: {e}")
        return False

def retrieve_context(query: str, top_k: int = 2) -> List[str]:
    """
    Takes a user query, embeds it, searches the Faiss index, and returns the top_k relevant text chunks.
    """
    if VECTOR_INDEX is None or EMBEDDER is None:
        return ["Error: Knowledge base not indexed or embedder failed to load."]

    # 1. Embed the query
    query_vector = EMBEDDER.encode([query], convert_to_numpy=True)
    
    # 2. Search the index (D is distances, I is indices)
    # The search will find the `top_k` closest vectors (lowest distance)
    distances, indices = VECTOR_INDEX.search(query_vector, top_k)
    
    # 3. Retrieve the corresponding text chunks
    retrieved_chunks = [DOCUMENTS[i] for i in indices[0]]
    
    # 
    
    return retrieved_chunks

# Execute the indexing phase when the script is run or imported
load_and_index_documents()

# Example test (optional, for debugging):
#print("\n--- Retrieval Test ---")
#context = retrieve_context("What is the primary use of Metformin?")
#print(f"Retrieved Context:\n{context}")