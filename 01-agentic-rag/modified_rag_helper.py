# Import the original RAGBase
from rag_helper import RAGBase

# Create your own class that inherits from RAGBase
class DocumentRAG(RAGBase):
    """RAG adapted for documents with filename/content schema."""
    
    def search(self, query: str, num_results: int = 5):
        """Override search to work with document schema."""
        # Adjust boost for document fields
        boost_dict = {"content": 3.0, "filename": 0.5}
        return self.index.search(
            query=query,
            num_results=num_results,
            boost_dict=boost_dict
        )
    
    def build_context(self, results):
        """Override context building for document schema."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Document {i} (from {result['filename']}):")
            context_parts.append(result['content'])
            context_parts.append("-" * 50)
        return "\n".join(context_parts)
    
    def rag(self, query: str, num_results: int = 5):
        """Override rag to return usage information."""
        # Reuse the base class's search and context building
        results = self.search(query, num_results=num_results)
        context = self.build_context(results)
        
        # Build prompt (same as base)
        prompt = f"""You're a course teaching assistant. Answer the question based on the context.

            Context:
            {context}

            Question: {query}

            Answer:"""
        
        # Send to LLM and capture full response
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # or your model
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract answer
        answer = response.choices[0].message.content
        
        # Extract usage
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return answer, usage