from typing import List, Optional, Tuple
from .google_drive_client import GoogleDriveClient
from .vector_db_manager import VectorDBManager
from .openai_client import OpenAIClient
from .rag_components import RAGDocument

class RAGSystem:
    """Main RAG system that integrates Google Drive, vector database, and OpenAI."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.drive_client = GoogleDriveClient()
        self.db_manager = VectorDBManager()
        self.openai_client = OpenAIClient()
    
    def load_documents_from_drive(self, folder_id: Optional[str] = None) -> List[RAGDocument]:
        """Load documents from Google Drive.
        
        Args:
            folder_id: Optional folder ID to load documents from
            
        Returns:
            List of loaded documents
        """
        return self.drive_client.load_documents(folder_id)
    
    def add_documents_to_db(self, documents: List[RAGDocument]):
        """Add documents to the vector database.
        
        Args:
            documents: List of documents to add
        """
        self.db_manager.add_documents(documents)
        self.db_manager.save()
    
    def query(self, query: str, k: int = 5) -> Tuple[str, List[Tuple[RAGDocument, float]]]:
        """Process a query and generate a response.
        
        Args:
            query: The user's query
            k: Number of similar documents to retrieve
            
        Returns:
            Tuple of (generated_response, retrieved_documents)
        """
        # Search for similar documents
        results = self.db_manager.search(query, k)
        
        # Create context from retrieved documents
        context = "\n".join([doc.content for doc, _ in results])
        
        # Generate response using OpenAI
        response, _ = self.openai_client.generate_response(
            f"Context:\n{context}\n\nQuery:\n{query}"
        )
        
        return response, results
    
    def update_database(self, folder_id: Optional[str] = None):
        """Update the vector database with new documents from Google Drive.
        
        Args:
            folder_id: Optional folder ID to update from
        """
        # Load new documents
        documents = self.load_documents_from_drive(folder_id)
        
        # Add to database
        self.add_documents_to_db(documents)
        
        return len(documents)

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Update database with documents from Google Drive
    print("Loading documents from Google Drive...")
    num_docs = rag_system.update_database()
    print(f"Loaded {num_docs} documents")
    
    # Test query
    query = "what is a vector store"
    print(f"\nProcessing query: {query}")
    
    response, results = rag_system.query(query)
    
    print("\nGenerated Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    print("\nRetrieved Documents:")
    for doc, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"Content: {doc.content}")
        print("-" * 80) 