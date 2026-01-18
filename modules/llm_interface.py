import logging
import config

logger = logging.getLogger(__name__)

# Embedding Interface
def get_embedding_model():
    """
    Returns the embedding model configured in config.py
    """
    logger.info("Using embedding model from config")
    return config.embed_model

# LLM Interface
def get_llm():
    """
    Returns the LLM configured in config.py.
    """
    logger.info("Using LLM from config")
    return config.llm

def change_llm_provider(new_provider: str):
    """
    Change LLM provider at runtime (openai / gemini).
    """
    if new_provider not in ("openai", "gemini"):
        raise ValueError("Invalid provider")
    
    config.LLM_PROVIDER = new_provider
    logger.info(f"LLM provider changed to : {new_provider}")
    

