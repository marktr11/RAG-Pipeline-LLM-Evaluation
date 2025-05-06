from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain_core.embeddings import Embeddings 
from typing import List

#This file contains the functions to create and populate the vector store

def create_and_populate_vector_store(
    documents: List[Document], embeddings: Embeddings
) -> InMemoryVectorStore:
    """
    Creates an in-memory vector store and populates it with the provided documents.

    Args:
        documents (List[Document]): The document chunks to add to the vector store.
        embeddings (Embeddings): The embedding model to use for vectorizing documents.

    Returns:
        InMemoryVectorStore: The populated vector store.
    """
    print(f"Creating vector store with {len(documents)} documents.")
    vector_store = InMemoryVectorStore(embeddings) 
    vector_store.add_documents(documents)
    print("Vector store created and populated.")
    return vector_store