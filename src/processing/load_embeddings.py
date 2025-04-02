#!/usr/bin/env python3
import argparse
from src.load_embeddings_to_db import load_embeddings_to_db

def main():
    parser = argparse.ArgumentParser(description="Load embeddings from files into PostgreSQL vector database")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Directory containing embeddings files")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents to process in each batch")
    
    args = parser.parse_args()
    
    try:
        doc_ids = load_embeddings_to_db(args.embeddings_dir, args.batch_size)
        print(f"\nSuccessfully loaded {len(doc_ids)} documents into the database")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 