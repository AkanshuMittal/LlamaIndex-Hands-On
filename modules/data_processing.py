import json 
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from modules.llm_interface import get_embedding_model
import config 

logger = logging.getLogger(__name__)

def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Splits the LinkedIn profile JSON data into nodes.
    
    Args:
        profile_data: LinkedIn profile data dictionary.
        
    Returns:
        List of document nodes.
    """
    try:
       profile_json = json.dumps(profile_data)  # LLM works with text data not dictionary

       document = Document(text=profile_json)   

       splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)

       nodes = splitter.get_nodes_from_documents([document])

       logger.info(f"Created {len(nodes)} nodes from profile data")

       return nodes
    
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []
      
def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Store the document chunks (nodes) in a vector database.
    
    Args: 
        nodes: List of document nodes to be indexed.
        
    Returns:
        VectorStoreIndex or None if indexing fails
    """

    try: 
        embedding_model = get_embedding_model()

        index = VectorStoreIndex(
            nodes = nodes,
            embed_model = embedding_model,
            show_progress=False
        )

        logger.info("Vector database created successfully.")

        return index
    
    except Exception as e:
        logger.error(f"Error in create_vector_datbase: {e}")
        return None

def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.
    
    Args:
        index: VectorStoreIndex to verify.
        
    Returns:
        True if all embeddings are vaild, False Otherwise.
    """

    try:
        vector_store = index._storage_context.vector_store   ## Internal vector store access
 
        node_ids = list(index.index_struct.nodes_dict.keys())  # It gives id for all nodes
       
        missing_embeddings = False
        for node_id in node_ids:
           embedding = vector_store.get(node_id)    ## Check each node embedding

           if embedding is None:
               logger.warning(f"Node ID {node_id} has a None embedding.")
               missing_embeddings=True
            
        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
        else:
            logger.info("All node embeddings are valid")
            return True

  
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False