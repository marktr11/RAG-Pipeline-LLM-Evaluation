from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config import PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP



def load_and_chunk_blog_content() -> List[Document]:
    """
    Loads content from the specified blog URL, extracts relevant sections,
    and splits it into smaller document chunks.

    Returns:
        List[Document]: A list of document chunks.
    """
    print(f"Loading content from: {PDF_FILE_PATH}")
    loader = PyPDFLoader(file_path=PDF_FILE_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) initially.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split content into {len(all_splits)} chunks.")
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