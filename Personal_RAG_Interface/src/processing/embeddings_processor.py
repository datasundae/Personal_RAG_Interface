from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import faiss
import json
import os
from src.config import GPU_CONFIG
from contextlib import nullcontext

class LocalEmbeddingsProcessor:
    """Handles the creation and storage of embeddings using local models."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """Initialize the embeddings processor with a local model."""
        # Use GPU configuration from config
        self.device = GPU_CONFIG["device"]
        print(f"\nUsing device: {self.device}")
        
        # Load the model and move it to the appropriate device
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Enable mixed precision if configured
        if GPU_CONFIG["use_mixed_precision"]:
            self.model.half()  # Convert model to FP16
            if GPU_CONFIG["use_amp"]:
                self.scaler = torch.cuda.amp.GradScaler()
        
        # Print model info
        if self.device == "mps":
            print("Using Apple Metal Performance Shaders (MPS)")
            print(f"Batch size: {GPU_CONFIG['batch_size']}")
            print(f"Mixed precision: {'Enabled' if GPU_CONFIG['use_mixed_precision'] else 'Disabled'}")
            print(f"Number of workers: {GPU_CONFIG['num_workers']}")
            print(f"Memory fraction: {GPU_CONFIG['memory_fraction']}")
        
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Initialize FAISS index
        self.index = None
        
    def create_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = None) -> np.ndarray:
        """Create embeddings for a list of chunks using batched processing."""
        if batch_size is None:
            batch_size = GPU_CONFIG["batch_size"]
            
        texts = [chunk["content"] for chunk in chunks]
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        embeddings = []
        with tqdm(total=total_batches, desc="Creating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Create embeddings with optimized settings
                with torch.cuda.amp.autocast() if GPU_CONFIG["use_amp"] else nullcontext():
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=self.device,
                        normalize_embeddings=True,  # Normalize for better similarity search
                        convert_to_tensor=True,  # Keep on GPU for faster processing
                        pin_memory=GPU_CONFIG["pin_memory"],
                        num_workers=GPU_CONFIG["num_workers"],
                        prefetch_factor=GPU_CONFIG["prefetch_factor"],
                        persistent_workers=GPU_CONFIG["persistent_workers"]
                    )
                embeddings.append(batch_embeddings)
                pbar.update(1)
        
        embeddings = np.vstack(embeddings)
        print(f"\nCreated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = "L2") -> None:
        """Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: numpy array of embeddings
            index_type: Type of FAISS index to create ("L2" or "IP" for inner product)
        """
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        else:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
            
        self.index.add(embeddings.astype('float32'))
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]], 
                       output_dir: str = "embeddings") -> None:
        """Save embeddings and chunks to disk.
        
        Args:
            embeddings: numpy array of embeddings
            chunks: List of chunks with metadata
            output_dir: Directory to save the files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        
        # Save chunks
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Save FAISS index if it exists
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(output_dir, "faiss.index"))
            
        print(f"Saved embeddings and chunks to {output_dir}")
    
    def load_embeddings(self, input_dir: str = "embeddings") -> tuple:
        """Load embeddings and chunks from disk.
        
        Args:
            input_dir: Directory containing the saved files
            
        Returns:
            tuple of (embeddings, chunks)
        """
        # Load embeddings
        embeddings = np.load(os.path.join(input_dir, "embeddings.npy"))
        
        # Load chunks
        with open(os.path.join(input_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Load FAISS index if it exists
        index_path = os.path.join(input_dir, "faiss.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        print(f"Loaded {len(embeddings)} embeddings and {len(chunks)} chunks")
        
        return embeddings, chunks
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar chunks using the FAISS index.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (score, chunk_index) tuples
        """
        if self.index is None:
            raise ValueError("No index available. Build or load an index first.")
            
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return list of (distance, index) tuples
        results = list(zip(distances[0], indices[0]))
        return results 