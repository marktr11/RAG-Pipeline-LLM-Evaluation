# RAG Pipeline and Evaluation Methodology 
This repository implements a simple RAG pipeline that reads from PDF files and answers question. Additionally, it proposes an evaluation architecture for the pipeline.


## Project Structure

-   `PART1_RAG_IMPLEMENTATION/`: Contains the Python code for the RAG pipeline.
    -   `PART1.md`: Documentation for Part 1, providing an overview of the RAG pipeline, its components, and how to run it.
-   `PART2_EVALUATION_METHODOLOGY/`: contain a document outlining the proposed system for evaluating RAG response quality.
    -   `Evaluation_methodology.md`
-   `exploration/` : contain the jupyter notebook file.
    - `rag_explore.ipynb` : learn and explore how to build RAG.
-   `README.md`: This file.

## Prerequisites

-   Python 3.5
-   `pip` for package installation
-   An OpenAI API Key (for embeddings and LLM generation)

## Part 1: RAG Pipeline

Consult : [PART1.md](https://github.com/marktr11/RAG-Pipeline-LLM-Evaluation/blob/master/PART1_RAG_IMPLEMENTATION/PART1.md)


### Expected Output (Part 1)

Question: What are the two main challenges that hinder the widespread application of the 'LLM-as-a-Judge' approach?

Generated answer: The two main challenges hindering the widespread application of the 'LLM-as-a-Judge' approach are the absence of a systematic review and concerns about reliability. The lack of formal definitions and fragmented understanding makes it difficult for researchers and practitioners to effectively apply this method. Additionally, issues related to the accuracy of evaluations and potential biases in the outputs raise significant reliability concerns.


## Part 2: Evaluation Methodology

The written component for Part 2, explaining the design and implementation of a RAG response evaluation system, can be found in the file:

