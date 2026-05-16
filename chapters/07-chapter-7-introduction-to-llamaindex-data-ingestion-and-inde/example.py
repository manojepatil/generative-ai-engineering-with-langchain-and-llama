import os
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.embeddings import resolve_embed_model

# 1. Define the problem: LLMs lack specific external knowledge.
# Imagine an LLM that only knows general facts, not details from this specific text.
# We need to give it access to specific information.

# 2. Prepare the data: This is our "external knowledge."
# In a real scenario, this could be
# a PDF, database, API, etc. Here, we use simple strings.
documents = [
    Document(text="The capital of France is Paris."),
    Document(text="Reinforcement learning is a type of machine learning "
                 "where an agent learns to make decisions by performing actions "
                 "in an environment and receiving rewards or penalties."),
    Document(text="LlamaIndex is a data framework for LLM applications that "
                 "provides tools for ingesting, structuring, and accessing "
                 "private or domain-specific data."),
    Document(text="LangChain is a framework for developing "
                 "applications powered by language models.")
]

print("--- Step 1: Ingesting Data ---")
print(f"Number of documents to ingest: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"  Document {i+1}: '{doc.text[:70]}...'")

# 3. Choose an embedding model (for converting text to numerical vectors).
# This is crucial for comparing text similarity.
# We'll use a local embedding model for this example to avoid external services.
# Make sure to install `transformers` and `torch` (or `tensorflow` for cpu)
# for `HuggingFaceEmbedding`.
try:
    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    # For a quicker test, if BGE download is slow, you can use:
    # embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
except ImportError:
    print("\nWARNING: `transformers` or `torch` not found for local embedding model.")
    print("Please install them: `pip install transformers torch`.")
    print("Proceeding without specific embedding model, which might fail or revert.")
    # Fallback or allow default behavior if package isn't there.
    # In a real project, you'd handle this more robustly.
    embed_model = None

# If embed_model is None, LlamaIndex might use a default or raise an error.
# For this example, we'll ensure we have one, even if it's a dummy for demonstration.
if embed_model is None:
    print("Using a placeholder embedding model for demonstration purposes due to "
          "missing dependencies. Retrieval quality may be affected.")
    from llama_index.core.embeddings import Embedding
    class DummyEmbedModel(Embedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            return [0.0] * 384  # A dummy vector of common dimension
        def _get_text_embedding(self, text: str) -> list[float]:
            return [0.0] * 384
    embed_model = DummyEmbedModel()


# 4. Create an index: This is where LlamaIndex structures and transforms the data.
# The `VectorStoreIndex` creates embeddings for each document and stores them
# in a vector store, making them searchable based on similarity.
print("\n--- Step 2: Indexing Data (creating vector embeddings) ---")
# If you want to persist the index:
# from llama_index.core.indices.vector_store import VectorStoreIndex
# from llama_index.core.vector_stores import SimpleVectorStore
# vector_store = SimpleVectorStore()
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )

# For this example, we keep it in memory for simplicity.
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
print("Data indexed successfully into a VectorStoreIndex.")

# 5. Create a query engine: This allows us to retrieve relevant information.
print("\n--- Step 3: Creating a Query Engine ---")
query_engine = index.as_query_engine()
print("Query engine created.")

# 6. Query the index: Ask a question and get relevant information back.
# This simulates the "Retrieval" part of RAG.
print("\n--- Step 4: Querying the Index (Retrieval) ---")
query = "What is LlamaIndex?"
print(f"Querying with: '{query}'")
response = query_engine.query(query)

print("\n--- Step 5: Displaying Retrieved Information ---")
print(f"Response from query engine: {response}")

# The response object often contains source nodes from which the answer was derived.
print("\n--- Step 6: Showing Source Nodes (Evidence) ---")
if response.source_nodes:
    print(f"Found {len(response.source_nodes)} relevant source nodes:")
    for i, node in enumerate(response.source_nodes):
        print(f"  Source Node {i+1} (score: {node.score:.2f}):")
        print(f"    Content: '{node.text[:100]}...'")
else:
    print("No source nodes found for this query.")

# This example demonstrates how LlamaIndex ingests raw text,
# converts it into structured representations (embeddings),
# stores them in an index, and enables retrieval based on semantic similarity.
# This retrieved information can then be passed to an LLM
# for generating a more informed answer (the "Generation" part of RAG).
