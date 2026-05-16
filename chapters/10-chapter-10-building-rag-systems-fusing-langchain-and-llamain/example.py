# rag_fusion.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllamaLLM
import os

# --- LlamaIndex Setup for Data Ingestion and Indexing ---
# Create a dummy data directory and file for demonstration
os.makedirs("data", exist_ok=True)
with open("data/policy.txt", "w") as f:
    f.write("Our company policy states that employees are eligible for 20 days of paid time off per year. Sick leave is separate and accrues at 1 day per month.")

# Load documents from the 'data' directory
documents = SimpleDirectoryReader("data").load_data()

# Initialize LlamaIndex's Ollama LLM and Embedding Model
# Ensure Ollama is running and has 'llama2' and 'nomic-embed-text' models pulled
llm_llama_index = LlamaIndexOllamaLLM(model="llama2", request_timeout=30.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Create a VectorStoreIndex from the documents using the specified embedding model
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create a LlamaIndex query engine (retriever)
query_engine = index.as_query_engine(llm=llm_llama_index, similarity_top_k=2)

# --- LangChain Setup for Orchestration and Generation ---
# Initialize LangChain's Ollama LLM
llm_langchain = Ollama(model="llama2")

# Define the RAG prompt template
template = """
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the LangChain RAG chain: retrieve -> format -> generate
rag_chain = (
    {"context": lambda x: query_engine.query(x["question"]).response, "question": RunnablePassthrough()}
    | prompt
    | llm_langchain
    | StrOutputParser()
)

# --- Example Usage ---
if __name__ == "__main__":
    question = "How many paid time off days do employees get per year?"
    
    print(f"User Question: {question}\n")
    
    # Invoke the RAG chain
    response = rag_chain.invoke({"question": question})
    
    print(f"Generated Answer: {response}")

    question_no_context = "What is the capital of France?"
    print(f"\nUser Question: {question_no_context}\n")
    response_no_context = rag_chain.invoke({"question": question_no_context})
    print(f"Generated Answer (without relevant context): {response_no_context}")
