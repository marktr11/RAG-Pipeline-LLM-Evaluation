

import os
from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

#modules
from processing import load_and_chunk_content, add_section_metadata
from vector_store import create_and_populate_vector_store
from step import analyze_query_step, retrieve_step, generate_step
from config import RAG_PROMPT_HUB_ID
import config



def run_rag_pipeline(question: str):
    """
    Sets up and runs the RAG application for a given question.
    """
    
    if not os.environ.get("OPENAI_API_KEY"):
       os.environ["OPENAI_API_KEY"] = config.LLM_API_KEY

    
    try:
        embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0) #temperature=0 for reproducible results
    except Exception as e:
        print(f"Error initializing OpenAI models: {e}")
        return

    # --- Splitting document ---
    doc_splits = load_and_chunk_content()
    doc_splits_with_metadata = add_section_metadata(doc_splits)

    # --- vetor store
    vector_store = create_and_populate_vector_store(
        documents=doc_splits_with_metadata, embeddings=embeddings_model
    )

    # --- pre-defined prompt
    rag_prompt = hub.pull(RAG_PROMPT_HUB_ID)

    print(f"\nRunning RAG Pipeline for question: '{question}'")

    # Query Analysis
    structured_query_obj = analyze_query_step(question=question, llm=llm)

    # Retrieve
    retrieved_documents = retrieve_step(
        structured_query=structured_query_obj, vector_store=vector_store
    )

    # Generation
    final_answer = generate_step(
        question=question,
        context_docs=retrieved_documents,
        llm=llm,
        prompt=rag_prompt,
    )
    
    #print results
    print("\n---RAG Pipeline Result ---")
    print(f"Original Question: {question}")
    print(f"Structured Query: {structured_query_obj}") 

    if retrieved_documents:
        print(f"Retrieved {len(retrieved_documents)} documents.")
    else:
        print("No documents retrieved.")
    print(f"Generated Answer: {final_answer}")


if __name__ == "__main__":
    user_question = "What are the two main challenges that hinder the widespread application of the 'LLM-as-a-Judge' approach?" 
    run_rag_pipeline(user_question)