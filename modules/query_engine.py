import logging 
from typing import Any

from llama_index.core import VectorStoreIndex, PromptTemplate
from modules.llm_interface import get_llm
import config

logger = logging.getLogger(__name__)

def is_greeting(query: str) -> bool:
    greetings = [
        "hi", "hello", "hey",
        "good morning", "good evening", "good afternoon"
    ]
    return query.lower().strip() in greetings

# GENERATE INITIAL FACTS
def generate_initial_facts(index: VectorStoreIndex) -> str:
    """
    Generate 3 interesting facts about the person's career or education.
    """
    try:
        llm = get_llm()

        facts_prompt = PromptTemplate(
            template = config.INITIAL_FACTS_TEMPLATE
        )

        query_engine = index.as_query_engine(
            streaming=False,   # Use for chat UI or typing effect need
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=llm,
            text_qa_template = facts_prompt  # How to use retrieved context 
        )

        query = (
            "Provide three interesting facts about this person's"
            "career or education."
        )

        response = query_engine.query(query)

        # response = {
        #         "response": "Final answer text",
        #         "source_nodes": [...],
        #         "metadata": {...}
        #     }
        
        return response.response  # return final text 
    
    except Exception as e:
        logging.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."
    
# ANSWER USER QUERY 
def answer_user_query(index: VectorStoreIndex, user_query: str) ->Any:
    """
    Answers the user's question using the vector database and the LLM.
    Handles greetings and irrelevant questions safely.
    """

    try:
        if is_greeting(user_query):
            return "Hello! You can ask me questions about this profile 😊"
        
        llm = get_llm() 

        question_prompt = PromptTemplate(
            template=config.USER_QUESTION_TEMPLATE
        )

        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=llm,
            text_qa_template = question_prompt
        )

        response = query_engine.query(user_query)
        
        # IRRELEVANT / NO CONTEXT
        if not response.response or "I don't know" in response.response:
            return (
                "I don't know. This information is not available "
                "in the provided profile."
            )

        return response.response # return final text
    
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Sorry, I couldn't process your question right now."
