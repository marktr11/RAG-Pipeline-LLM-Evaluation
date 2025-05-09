from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain_core.embeddings import Embeddings 
from typing import List
import logging

def create_and_populate_vector_store(
    documents: List[Document], embeddings: Embeddings
) -> InMemoryVectorStore:
    """
    Creates an in-memory vector store and populates it with the provided documents.

    1. Storing documents (Indexing)

    Args:
        documents (List[Document]): The document chunks to add to the vector store.
        embeddings (Embeddings): The embedding model to use for vectorizing documents.

    Returns:
        InMemoryVectorStore: The populated vector store.
    """
    logger = logging.getLogger(__name__)
    #number of documents to be added to the vector store
    logger.info(f"Creating vector store with {len(documents)} chunks.")

    # Create a new in-memory vector store using the provided embedding model
    vector_store = InMemoryVectorStore(embeddings) 
    # Add the documents to the vector store (embedding happens internally)
    vector_store.add_documents(documents)

    logger.info("Vector store created and populated.")

    return vector_store

# For future scalability with a larger number of documents/sources,
# consider replacing InMemoryVectorStore with a persistent and more efficient
# vector database (e.g., FAISS, ChromaDB, or a cloud-based solution).
# InMemoryVectorStore re-processes documents on each run, which is inefficient for larger datasets.