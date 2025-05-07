

import os
from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Custom modules for RAG pipeline components
from processing import load_and_split_content, add_section_metadata
from vector_store import create_and_populate_vector_store
from step import analyze_query_step, retrieve_step, generate_step
from config import RAG_PROMPT_HUB_ID
import config

# --- Setup logging ---
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Suppress noisy logs from third-party libraries
for noisy_logger in ["httpx", "openai", "langchain", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)



def run_rag_pipeline(question: str):
    """
    Runs the full Retrieval-Augmented Generation (RAG) pipeline for a given input question.
    This includes loading and splitting documents, vectorizing them, analyzing the query,
    retrieving relevant chunks, and generating the final answer using an LLM.
    """
    
    # Ensure the OpenAI API key is set in environment variables
    if not os.environ.get("OPENAI_API_KEY"):
       os.environ["OPENAI_API_KEY"] = config.LLM_API_KEY

    
    try:
        # Initialize OpenAI models for embedding and LLM
        embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0) #temperature=0 for reproducible results
    except Exception as e:
        logging.error(f"Error initializing OpenAI models: {e}")
        return

    # --- Load and Splitting document ---
    doc_splits = load_and_split_content() # Load and split PDF into chunks
    doc_splits_with_metadata = add_section_metadata(doc_splits) # Add section metadata

    # --- Vector Store Creation
    vector_store = create_and_populate_vector_store(
        documents=doc_splits_with_metadata, embeddings=embeddings_model
    )

    # --- Prompt Retrieval
    rag_prompt = hub.pull(RAG_PROMPT_HUB_ID) # Load pre-defined prompt template from LangChain Hub

    print(f"\nRunning RAG Pipeline for question: '{question}'\n")

    # --- Query Analysis ---
    structured_query_obj = analyze_query_step(question=question, llm=llm) # Convert question into structured format

    # --- Document Retrieval ---
    retrieved_documents = retrieve_step(
        structured_query=structured_query_obj, vector_store=vector_store
    )

    # --- Answer Generation ---
    final_answer = generate_step(
        question=question,
        context_docs=retrieved_documents,
        llm=llm,
        prompt=rag_prompt,
    )
    
    #print results
    print("\n---RAG Pipeline Result ---")
    print(f"Question: {question}")
    print(f"\nGenerated Answer: {final_answer}")



# Entry point when this script is executed directly
if __name__ == "__main__":
    user_question = "What are the two main challenges that hinder the widespread application of the 'LLM-as-a-Judge' approach?" 
    run_rag_pipeline(user_question)