import torch
from sentence_transformers import SentenceTransformer
from src.rag import RAGSystem
import os
from collections import defaultdict

def deduplicate_results(results):
    """Remove duplicate results based on text content."""
    seen_texts = set()
    unique_results = []
    
    for doc in results:
        # Normalize text for comparison
        text = doc['text'].strip().lower()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_results.append(doc)
    
    return unique_results

def format_text(text, max_length=500):
    """Format text for display with proper sentence breaks."""
    # Split into sentences
    sentences = text.split('. ')
    
    # Build formatted text
    formatted_text = ''
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            break
        formatted_text += sentence + '. '
        current_length += len(sentence)
    
    return formatted_text.strip()

def main():
    # Initialize model with GPU support
    device = torch.device("mps")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model.to(device)
    
    # Initialize RAG system
    book_name = "The Berklee Book of Jazz Harmony-1"
    embeddings_dir = f"embeddings/{book_name}"
    
    rag = RAGSystem(
        embeddings_dir=embeddings_dir,
        model=model,
        device=device
    )
    
    # Test queries
    test_queries = [
        "What are the basic elements of jazz harmony?",
        "How are seventh chords constructed?",
        "What is modal harmony in jazz?",
        "Explain chord voicings in jazz",
        "What are chord tensions and how are they used?"
    ]
    
    # Run queries
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 80)
        
        # Get results and deduplicate
        results = rag.search(query, k=5)  # Get more results to account for deduplication
        unique_results = deduplicate_results(results)
        
        # Group results by page
        results_by_page = defaultdict(list)
        for doc in unique_results:
            page = doc.get('metadata', {}).get('page', 'Unknown')
            results_by_page[page].append(doc)
        
        # Display results
        for i, doc in enumerate(unique_results[:3], 1):
            print(f"\nResult {i} (Score: {doc['score']:.4f})")
            if 'metadata' in doc and 'page' in doc['metadata']:
                print(f"Page: {doc['metadata']['page']}")
            print("-" * 40)
            print(format_text(doc['text']))
            print()

if __name__ == "__main__":
    main() 