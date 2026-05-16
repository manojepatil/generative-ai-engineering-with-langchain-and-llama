import random

# --- Self-Correction in RAG (Simplified Example) ---

class DocumentStore:
    """Simulates a document store for Retrieval Augmented Generation (RAG)."""
    def __init__(self):
        self.documents = {
            "doc1": "The capital of France is Paris and it's known for the Eiffel Tower.",
            "doc2": "The capital of Germany is Berlin. Berlin has a rich history.",
            "doc3": "London is the capital of the United Kingdom.",
            "doc4": "The Eiffel Tower is in Paris, a city in France.",
            "doc5": "A different random fact about something unrelated."
        }

    def retrieve(self, query: str) -> list[str]:
        """Retrieves relevant documents based on a simple keyword match."""
        relevant_docs = []
        query_words = query.lower().split()
        for doc_id, content in self.documents.items():
            if any(word in content.lower() for word in query_words):
                relevant_docs.append(content)
        return relevant_docs if relevant_docs else [self.documents["doc5"]] # Fallback

class LLM:
    """Simulates a Large Language Model for generation and self-reflection."""
    def generate(self, prompt: str) -> str:
        """Generates a response based on the prompt."""
        # A simple, rule-based "LLM" for demonstration
        if "capital of France" in prompt:
            if "Paris" in prompt:
                return "The capital of France is indeed Paris."
            else:
                return "Based on the information, the capital of France is Paris."
        elif "capital of Germany" in prompt:
            if "Berlin" in prompt:
                return "Yes, Berlin is the capital of Germany."
            else:
                return "The capital of Germany is Berlin."
        elif "correct the following" in prompt and "Incorrect: The capital of France is London." in prompt:
            return "Correction: The capital of France is Paris."
        return "I'm not sure about that specific query without more context."

    def reflect_and_correct(self, original_query: str, retrieved_docs: list[str], generated_answer: str) -> bool:
        """
        Simulates the LLM reflecting on its answer and retrieved docs.
        Returns True if a correction is needed, False otherwise.
        """
        # For simplicity, we trigger correction if the initial answer seems off
        # or if specific keywords are missing in the generated answer despite being in docs.
        
        if "capital of France" in original_query and "Paris" not in generated_answer and "Paris" in "".join(retrieved_docs):
            print(f"    [Reflection Agent]: Detected potential inaccuracy. Retrying with a more focused prompt.")
            return True
        if "capital of Germany" in original_query and "Berlin" not in generated_answer and "Berlin" in "".join(retrieved_docs):
            print(f"    [Reflection Agent]: The answer seems incomplete. Re-evaluating.")
            return True
        return False

# --- Main RAG Loop with Self-Correction ---
def self_correcting_rag(query: str, doc_store: DocumentStore, llm: LLM, max_retries: int = 2) -> str:
    """
    Demonstrates a simplified self-correcting RAG pipeline.
    If the initial generation is deemed insufficient, it tries to re-prompt.
    """
    print(f"\n--- Processing Query: '{query}' ---")
    retrieved_docs = doc_store.retrieve(query)
    print(f"  [Retrieval Agent]: Retrieved documents: {[doc[:30] + '...' for doc in retrieved_docs]}")

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  [Correction Cycle {attempt + 1}]")

        # Initial prompt combining query and context
        context = "\n".join(retrieved_docs)
        prompt = f"Based on the following information, answer the question: '{query}'\n\nInformation:\n{context}\n\nAnswer:"
        
        generated_answer = llm.generate(prompt)
        print(f"  [Generation Agent]: Attempt {attempt + 1} - Generated answer: '{generated_answer}'")

        if llm.reflect_and_correct(query, retrieved_docs, generated_answer):
            if attempt < max_retries - 1:
                # In a real system, the reflection agent might reformulate the query,
                # refine retrieved docs, or generate a correction prompt.
                # Here, we simulate a simple re-prompt with a stronger instruction
                # or focus on missing info.
                retrieved_docs_for_correction = doc_store.retrieve(query + " specific capital") # Simulate refined retrieval
                retrieved_docs = retrieved_docs_for_correction # Use refined docs for next attempt
                print(f"  [RAG System]: Self-correction triggered. Retrying...")
            else:
                print(f"  [RAG System]: Max retries reached. Returning current best answer.")
                return generated_answer
        else:
            print(f"  [RAG System]: No correction needed or max retries reached. Final answer.")
            return generated_answer

    return generated_answer # Should not be reached if max_retries is handled correctly

if __name__ == "__main__":
    doc_store = DocumentStore()
    llm = LLM()

    # Example 1: Successful retrieval and generation
    final_answer = self_correcting_rag("What is the capital of France?", doc_store, llm)
    print(f"Final Answer: {final_answer}\n")

    # Example 2: Query that might trigger a correction due to simplified LLM logic
    # Assume our simple LLM initially doesn't perfectly generate "Paris" if only "France" is present.
    # The reflection agent catches this and guides it to refine.
    final_answer = self_correcting_rag("Tell me about the capital of France.", doc_store, llm)
    print(f"Final Answer: {final_answer}\n")

    # Example 3: Another correction scenario for Germany
    final_answer = self_correcting_rag("Where is Germany's capital?", doc_store, llm)
    print(f"Final Answer: {final_answer}\n")

    # Example 4: A query our simplified system can't answer well, showing the limits
    final_answer = self_correcting_rag("What is the best food in the world?", doc_store, llm)
    print(f"Final Answer: {final_answer}\n")
