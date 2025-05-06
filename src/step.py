
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import BasePromptTemplate
from langchain_core.documents import Document 
from typing import List

from state_schema import Search 

def analyze_query_step(question: str, llm: BaseLanguageModel) -> Search:
    """
    Analyzes the user's question using an LLM to produce a structured Search object.
    """
    print(f"--- Step: analyze_query for question: '{question}' ---")
    structured_llm = llm.with_structured_output(Search)
    query_structured = structured_llm.invoke(question)
    print(f"Structured query: {query_structured}")
    return query_structured 


def retrieve_step(structured_query: Search, vector_store: VectorStore) -> List[Document]:
    """
    Retrieves relevant documents from the vector store based on the structured query.
    """
    print(f"--- Step: retrieve for structured query: {structured_query} ---")
    

    retrieved_docs = vector_store.similarity_search(
        structured_query["query"],
    )
    print(f"Retrieved {len(retrieved_docs)} documents for section '{structured_query['section']}'.")
    return retrieved_docs 


def generate_step(
    question: str, context_docs: List[Document], llm: BaseLanguageModel, prompt: BasePromptTemplate
) -> str:
    """
    Generates an answer using the LLM based on the question and retrieved context.
    """
    print(f"--- Step: generate for question: '{question}' ---")
    if not context_docs:
        print("Warning: No context provided for generation. LLM may answer from general knowledge.")
        docs_content = "No specific context found." #
    else:
        docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    
    
    
    messages = prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    generated_answer = response.content
    print(f"Generated answer: {generated_answer}")
    return generated_answer 