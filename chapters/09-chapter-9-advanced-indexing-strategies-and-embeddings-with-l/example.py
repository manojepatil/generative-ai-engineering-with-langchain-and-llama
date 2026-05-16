import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.mock import MockLLM
from llama_index.core.settings import Settings
from llama_index.embeddings.mock import MockEmbedding
import chromadb

# --- Mock Setup (no external services) ---
# Create a dummy directory and file for testing
if not os.path.exists("data"):
    os.makedirs("data")
with open("data/document.txt", "w") as f:
    f.write("LlamaIndex helps build LLM applications over custom data. "
            "It provides indexing and querying capabilities. "
            "Vector stores are crucial for semantic search.")

# Configure LlamaIndex to use mock LLM and embedding for local execution
Settings.llm = MockLLM()
Settings.embed_model = MockEmbedding(embed_dim=1536) # Default OpenAI embed_dim

# --- Step 1: Load documents ---
# Load documents from a directory. This simulates reading your data.
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document(s).")

# --- Step 2: Initialize a Vector Store (ChromaDB in-memory for this example) ---
# Vector stores are a common indexing strategy for semantic search.
# ChromaDB is a popular choice and can run in-memory for testing.
db = chromadb.PersistentClient(path="./chroma_db") # Stores data on disk
chroma_collection = db.get_or_create_collection("my_documents_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
print(f"Initialized ChromaDB vector store.")

# --- Step 3: Create a Storage Context ---
# The StorageContext manages persistence of various index components.
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Step 4: Create a VectorStoreIndex ---
# This step embeds the documents and stores them in the configured vector store.
print("Creating VectorStoreIndex. This will embed the documents...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("VectorStoreIndex created successfully.")

# --- Step 5: Query the index ---
# Now we can query the index to retrieve relevant information based on semantic similarity.
print("\nQuerying the index...")
query_engine = index.as_query_engine()
response = query_engine.query("What are the key capabilities of LlamaIndex?")
print(f"Query: What are the key capabilities of LlamaIndex?")
print(f"Response: {response}")

# --- Clean up (optional) ---
# Remove the dummy data and chroma db files
os.remove("data/document.txt")
os.rmdir("data")
# In production, you'd manage your ChromaDB persistence more carefully.
# For this example, we demonstrate clearing it for a fresh run if needed.
# db.delete_collection("my_documents_collection") # Uncomment to delete collection

# This example demonstrates the core idea:
# 1. Loading data.
# 2. Choosing and configuring an indexing strategy (Vector Store with ChromaDB).
# 3. Building an index using an embedding model (mocked here).
# 4. Querying the index to retrieve relevant information.
