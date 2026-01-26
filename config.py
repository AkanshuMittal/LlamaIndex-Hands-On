import os 
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
#from llama_index.llms.gemini import Gemini
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

if LLM_PROVIDER == "openai":
    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )

    embed_model = OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY")
    )

# elif LLM_PROVIDER == "gemini":
#     llm = Gemini(
#         model="models/gemini-pro",
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.3
#     )

#     embed_model = HuggingFaceEmbedding(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

else:
    raise ValueError("Invalid LLM_PROVIDER. Use 'openai' or 'gemini'.")


CHUNK_SIZE = 400
SIMILARITY_TOP_K = 5

## Prompt Template

INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer using only the information provided in the context.
If the answer is not available, say:
"I don't know. This information is not available in the provided profile."
"""