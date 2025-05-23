from typing import Literal, List
from typing_extensions import Annotated, TypedDict
from langchain_core.documents import Document

class Search(TypedDict):
    """
    Structured representation of a search query, including:
    - query: The natural language search query to execute.
    - section: The target section of the document where the search should be performed.
    
    This class represents a search with an additional 'section' field to focus the search
    on specific parts of the document (e.g., 'beginning', 'middle-1', 'middle-2', 'end').
    """
    query: Annotated[str, "The natural language search query to execute."]  # Search query in natural language
    section: Annotated[
    Literal["beginning", "middle-1", "middle-2", "end"],  # Added one more section ('middle-2') to better facilitate information retrieval.
    "The specific section of the document to target for the search."
    ]


