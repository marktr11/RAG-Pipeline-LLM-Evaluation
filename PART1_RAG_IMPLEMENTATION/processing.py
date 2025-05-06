from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config import PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)
def load_and_split_content() -> List[Document]: 
    """
    Loads content from the specified pdf file, extracts relevant sections,
    and splits it into smaller document chunks.

    1. Loading document
    2. Document splitting

    Returns:
        List[Document]: A list of document chunks.
    """
    # Initialize a PDF loader with the specified file path
    loader = PyPDFLoader(file_path=PDF_FILE_PATH)
    # Load all pages from the PDF into a list of Document objects
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} document(s) / page(s).")
    
    # Initialize a text splitter to break large documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Max characters in one chunk
        chunk_overlap=CHUNK_OVERLAP # Overlap between chunks to preserve context
    )

    # Apply the text splitter to the loaded documents
    # `all_splits` is a list of Document objects, where each object represents a chunk of text
    # taken from the original PDF. These chunks are designed to be small enough to be processed
    # efficiently by LLMs (e.g., GPT), while maintaining enough context through overlapping text.
    all_splits = text_splitter.split_documents(docs)

    logger.info(f"Split content into {len(all_splits)} chunks.")

    return all_splits


def add_section_metadata(documents: List[Document]) -> List[Document]:
    """
    Adds a 'section' metadata field ('beginning', 'middle', 'end') to each document
    based on its position in the list.

    Args:
        documents (List[Document]): The list of document chunks.

    Returns:
        List[Document]: The list of document chunks with added 'section' metadata.
    """
    total_documents = len(documents)
    if total_documents == 0:
        return documents
        
    third = total_documents // 3

    for i, document in enumerate(documents):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"
    print("Added 'section' metadata to document chunks.")
    return documents