from llama_index.readers.file import PDFReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.database import CassandraReader, CassandraReaderTableSchema, CassandraReaderTableQuery
from llama_index.readers.schema import Document

# Simulate a PDF file (in a real scenario, this would be a path to a PDF)
# For demonstration, we'll create a dummy document.
# In a real application:
# pdf_documents = PDFReader().load_data(file=Path("./data/example.pdf"))
pdf_docs = [Document(text="This is content from a simulated PDF document about LlamaIndex features.")]

# Simulate fetching data from a web page
# In a real application:
# web_documents = SimpleWebPageReader(html_to_text=True).load_data(urls=["https://www.llamaindex.ai/"])
web_docs = [Document(text="This is content from a simulated web page detailing LlamaIndex's architecture.")]


# Simulate loading data from a Cassandra database
# For demonstration, we'll create dummy documents.
# In a real application, you'd configure CassandraReader similar to:
# cassandra_reader = CassandraReader(
#     session=cass_session,
#     keyspace="my_keyspace",
#     table="my_table",
#     metadata_extractor=MyMetadataExtractor()
# )
# cassandra_table_data = cassandra_reader.load_data(
#     table_schema=CassandraReaderTableSchema(
#         table_name="my_llama_data",
#         column_names=["id", "text_content", "created_at"]
#     )
# )
cassandra_docs = [
    Document(text="This record from Cassandra contains customer support ticket #12345: 'LLM not responding.'"),
    Document(text="Another record from Cassandra with product review: 'LlamaIndex helped us speed up development significantly!'")
]

# Unify all collected documents
all_documents = []
all_documents.extend(pdf_docs)
all_documents.extend(web_docs)
all_documents.extend(cassandra_docs)

# Print out the content of the unified documents to demonstrate ingestion
print("--- Unified Documents from Various Sources ---")
for i, doc in enumerate(all_documents):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.text[:100]}...") # Print first 100 chars
    print(f"Metadata: {doc.metadata}")

print(f"\nTotal documents collected: {len(all_documents)}")
print("\nThis demonstrates loading data from diverse sources into a unified Document format for LlamaIndex.")
