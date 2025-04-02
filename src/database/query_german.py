from src.embeddings import load_embeddings_and_index
from src.rag import RAGSystem
import torch
from sentence_transformers import SentenceTransformer

def query_german_text(query: str, top_k: int = 3):
    """
    Query the German text using an English query.
    
    Args:
        query: English query string
        top_k: Number of results to return
    """
    # Load the embeddings and index
    embeddings_dir = "embeddings/Wirtschaft und Gesellschaft"
    embeddings, index, documents = load_embeddings_and_index(embeddings_dir)
    
    # Initialize model with Metal backend
    device = torch.device("mps")  # Metal Performance Shaders
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
    
    # Create RAG system
    rag = RAGSystem(embeddings, index, documents, model, device)
    
    # Perform the search
    results = rag.search(query, top_k=top_k)
    
    # Print results
    print(f"\nQuery: {query}\n")
    print("Results:")
    print("=" * 80)
    
    for i, (score, doc) in enumerate(results, 1):
        print(f"\nResult {i} (Score: {score:.4f}):")
        print("-" * 40)
        print(doc.text[:200] + "...")
        print("-" * 40)

if __name__ == "__main__":
    # Example queries
    queries = [
        "What does Weber say about capitalism and religion?",
        "How does Weber define social action?",
        "What are the different types of authority according to Weber?"
    ]
    
    for query in queries:
        query_german_text(query)
        print("\n" + "=" * 80 + "\n") 