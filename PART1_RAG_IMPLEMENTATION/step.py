
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import BasePromptTemplate
from langchain_core.documents import Document 
from typing import List
import logging
from state_search import Search 

logger = logging.getLogger(__name__)


def analyze_query_step(question: str, llm: BaseLanguageModel) -> Search:
    """
    Analyzes the user's question using a Language Learning Model (LLM) to produce a structured Search object.

    Args:
        question (str): The user's original question that needs to be analyzed.
        llm (BaseLanguageModel): The language model used to process the question and generate a structured query.

    Returns:
        Search: A structured Search object created by the LLM, representing the processed query.
    """
    # Log the current step in the pipeline, displaying the user's question being analyzed
    logger.info(f"Step: Query Analysis for question: '{question}'")
    
    # Use the language model to enable structured output by specifying the 'Search' type
    structured_llm = llm.with_structured_output(Search)
    # Use the structured language model to invoke the query analysis and produce a structured output
    query_structured = structured_llm.invoke(question)
    
    # log the structured query returned by the language model, showing the transformation of the question
    logger.info(f"Structured query: {query_structured}")

    return query_structured 


def retrieve_step(structured_query: Search, vector_store: VectorStore) -> List[Document]:
    """
    Retrieves relevant documents from the vector store based on the structured query.

    Args:
        structured_query (Search): The structured query object containing the search query and relevant section.
        vector_store (VectorStore): The vector store where the documents are indexed and stored.

    Returns:
        List[Document]: A list of relevant documents retrieved based on the structured query.
    """
    # Log the step in the pipeline, showing the structured query that is being used for retrieval
    logger.info(f"Step: retrieve for structured query: {structured_query}")
    
    # Perform a similarity search in the vector store using the 'query' field from the structured Search object
    # Note: The similarity search does not guarantee that the retrieved chunks will have the same length.
    # The returned documents (chunks) may vary in length because the vector store may split the text differently.
    retrieved_docs = vector_store.similarity_search(
        structured_query["query"],
    )
    # Log the number of documents retrieved and the section associated with the query
    logger.info(f"Retrieved {len(retrieved_docs)} documents for section '{structured_query['section']}'.")

    return retrieved_docs 


def generate_step(
    question: str, context_docs: List[Document], llm: BaseLanguageModel, prompt: BasePromptTemplate
) -> str:
    """
    Generates an answer using the LLM based on the question and retrieved context.

    Args:
        question (str): The user's input question.
        context_docs (List[Document]): List of context documents retrieved via similarity search.
        llm (BaseLanguageModel): The language model used to generate the response.
        prompt (BasePromptTemplate): The prompt template for formatting the input to the LLM.

    Returns:
        str: The generated answer from the LLM.
    """
    logger.info(f"Step: generate for question: '{question}'")

    # If no documents were retrieved, warn the user and provide a fallback context
    if not context_docs:
        print("Warning: No context provided for generation. LLM may answer from general knowledge.")
        docs_content = "No specific context found." # Placeholder context if nothing was retrieved
    else:
        # Combine the content of all retrieved documents into a single string, separated by double newlines
        docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    
    
    # Format the input using the prompt template by injecting the question and the context
    messages = prompt.invoke({
        "question": question, 
        "context": docs_content
    })

    # Call the LLM to generate a response using the formatted message
    response = llm.invoke(messages)
    # Extract the textual content of the generated response
    generated_answer = response.content
    print(f"\nGenerated answer: {generated_answer}")
    

    # Save result to file
    output_path = "PART1_RAG_IMPLEMENTATION\output_example.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{question}\n\n")
        f.write("\n--- Retrieved Context Documents ---\n")
        for i, doc in enumerate(context_docs):
            chunk_text = f"Document {i+1}:\n{doc.page_content}\n---\n"
            f.write(chunk_text)
        f.write("Generated Answer:\n")
        f.write(generated_answer)
    logger.info(f"\nOutput saved to {output_path}")

    return generated_answer 