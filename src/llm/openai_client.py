import os
import httpx
from typing import List, Dict, Any

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
    def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-4") -> str:
        """Get a chat completion from OpenAI"""
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None
            
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context for the LLM"""
        context_parts = []
        for doc in documents:
            context_parts.append(f"From {doc['metadata']['title']} by {doc['metadata']['author']} (Page {doc['metadata'].get('page', 'N/A')}):")
            context_parts.append(doc['text'])
            context_parts.append("")  # Add blank line between documents
        return "\n".join(context_parts) 