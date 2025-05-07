import os
from dotenv import load_dotenv
import logging

# basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# Absolute path to the directory containing the current script file (config.py)
config_dir = os.path.dirname(os.path.abspath(__file__))
#print(config_dir)

# Combine the config.py directory with the name of the .env file
env_path = os.path.join(config_dir, '.env')
#print(env_path)

# The key point is to have the absolute path to the .env file.
# Without it, when importing config from other folders, it may cause an empty value (since .env cannot be found).
load_dotenv(env_path)
#print(load_dotenv(env_path))

# Print the current working directory , root folder where you open the terminal.
current_working_directory = os.getcwd()
logging.info(f"Current working directory: {current_working_directory}")
# --- Essential Configurations --

# == LLM Configuration ==
# Load the LLM API key from the environment variable.
LLM_API_KEY = os.getenv("LLM_API_KEY_ENV")

# == LangSmith Configuration ==
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Standard environment variable to enable LangSmith tracing (set to "true" or "1")
# Example .env entry: LANGSMITH_TRACING_V2=true
LANGSMITH_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true" # Default to false if not set

# Optional: Specify a LangSmith project name
LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT") # Can be None if not set


# == Data Configuration ==
# Define the path to the input PDF document.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_FILENAME = "publication.pdf"
PDF_FILE_PATH = os.path.join(PROJECT_ROOT, "data", PDF_FILENAME)


# --- Configuration Loading Checks ---
# Inform the user about the status of essential configurations.

if not LLM_API_KEY:
    logging.warning("LLM API Key environment variable not found in .env")
else:
    # Never log the actual key! Just confirm it's loaded.
    logging.info("LLM API Key loaded from environment.")


# LangSmith is optional
if LANGSMITH_TRACING_ENABLED:
    if not LANGSMITH_API_KEY:
        logging.warning("LangSmith tracing is enabled (LANGSMITH_TRACING_V2=true), but LANGSMITH_API_KEY was not found in .env.")
    else:
        logging.info("LangSmith API Key loaded.")
        if LANGSMITH_PROJECT_NAME:
            logging.info(f"LangSmith project name set to: {LANGSMITH_PROJECT_NAME}")
        else:
            logging.info("LangSmith project name not specified (will use default).")
else:
    logging.info("LangSmith tracing is disabled (LANGSMITH_TRACING_V2 is not 'true').")


if not os.path.exists(PDF_FILE_PATH):
    logging.warning(f"PDF file not found at expected path: {PDF_FILE_PATH}")
else:
    logging.info(f"PDF file path configured: {PDF_FILE_PATH}")



# Text splitting parameters
# For documents >3500 characters, initially I set CHUNK_SIZE to 1000 and CHUNK_OVERLAP to 200 by default. 
# However, I found that the retrieved context seemed less reliable, as the chunks appeared to be either too short or lacked sufficient relevant content.
# Based on this, I increased the CHUNK_SIZE to 1200 and set the CHUNK_OVERLAP to 300 for better context retrieval.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


# Model names
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
LLM_MODEL_NAME = "gpt-4o-mini" 

# Prompt Hub ID
RAG_PROMPT_HUB_ID = "rlm/rag-prompt"