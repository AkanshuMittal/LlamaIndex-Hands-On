import logging 
from typing import Any

from llama_index.core import VectorStoreIndex, PromptTemplate
from modules.llm_interface import get_llm
import config

logger = logging.getLogger(__name__)

def generate_initial_facts(index: VectorStoreIndex) -> str:
    """
    Generate interesting facts about the person's career or education.
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
    
def answer_user_query(index: VectorStoreIndex, user_query: str) ->Any:
    """
    Answers the user's question using the vector database and the LLM.
    """

    try:
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

        answer = query_engine.query(user_query)
        return answer
    
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer"
