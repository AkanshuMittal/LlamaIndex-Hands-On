import sys  # used for stdout access purpose
import logging 
import uuid   # It is used for generation unique id for every user
import gradio as gr
import config

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data,create_vector_database,verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)  #  logs print in terminal
    ]
)

logger = logging.getLogger(__name__)

## Session store 
active_indices = {}   # session_id -> VectorStoreIndex

# Profile Processing function
def process_profile(linkedin_url, api_key, use_mock):
    """
    Used for processing the LinkedIn Profile.
    data -> chunks -> embeddings -> vector DB -> inital facts
    """

    try: 
        if use_mock and not linkedin_url:
             linkedin_url = "https://www.linkedin.com/in/sample-profile/"

             profile_data = extract_linkedin_profile(
             linkedin_profile_url=linkedin_url,
             api_key=api_key if not use_mock else None,
             mock=use_mock,
        )
        
        # if profile data not found 
        if not profile_data:
            return "Failed to retrieve profile data.", None
        
        ## DATA PROCESSING LAYER
        
        nodes = split_profile_data(profile_data)

        if not nodes:
            return "Failed to process profile data.", None
        
        ## VECTOR DATABASE LAYER

        index = create_vector_database(nodes)

        if not index:
            return "Failed to create vector database.", None
        
        ## Check embedding is made or not for each node
        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing.")

        ## Initial RAG query
        facts = generate_initial_facts(index)

        ## SESSION MANAGEMENT
        session_id = str(uuid.uuid4())  # Unique session id for each profile

        active_indices[session_id] = index  # Store index against session ID

        # return session id and facts to the user 
        return (
            f"Profile processed successfully!\n\n"
            f"Here are 3 interesting facts:\n\n{facts}",
            session_id,
        )
    
    except Exception as e:
        # Kisi bhi unexpected error ko log karna
        logger.error(f"Error in process_profile: {e}")
        return f"Error: {str(e)}", None


## CHAT FUNCTION
def chat_with_profile(session_id, user_query, chat_history):
    """
    Used for chat with processed file.
    """

    # if session id not found
    if not session_id:
        return chat_history + [[user_query, "Please process a profile first."]]

    # if session is expired
    if session_id not in active_indices:
        return chat_history + [[user_query, "Session expired. Please reprocess the profile."]]

    # if query is empty
    if not user_query.strip():
        return chat_history
    
    try:
        # find index for session id
        index = active_indices[session_id]  

        response = answer_user_query(index, user_query)

        return chat_history + [[user_query, response.response]]
    
    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return chat_history + [[user_query, f"Error: {str(e)}"]]
    
## Gradio UI 
def create_gradio_interface():
    """
    It defines the complete UI structure for Gradio.
    """

    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:

        # UI heading
        gr.Markdown("# 🔗 LinkedIn Icebreaker Bot")

        # Short description
        gr.Markdown(
            "Generate personalized icebreakers and ask questions "
            "using a Retrieval-Augmented Generation (RAG) pipeline."
        )

    ## Process Profile
    with gr.Tab("Process Profile"):
        with gr.Row():
            with gr.Column():
                # LinkedIn URL input
                linkedin_url = gr.Textbox(
                label="LinkedIn Profile URL",
                placeholder="https://www.linkedin.com/in/username/",
            )

                # ProxyCurl API key input 
                api_key = gr.Textbox(
                label="ProxyCurl API Key (optional)",
                type="password",
                placeholder="Only required for real LinkedIn data",
                value=config.PROXYCURL_API_KEY,
            )

                # Mock data toggle
                use_mock = gr.Checkbox(
                label="Use Mock Data",
                value=True,
            )

                 # Profile process button
                process_btn = gr.Button("Process Profile")

                with gr.Column():
                    # Output box for initial facts
                    result_text = gr.Textbox(label="Initial Facts", lines=10)

                    # Hidden session ID
                    session_id = gr.Textbox(visible=False)

            # Button click event binding
            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock],
                outputs=[result_text, session_id],
            )

        ## CODE FOR CHAT
        with gr.Tab("Chat"):
            gr.Markdown("### Ask questions about the processed LinkedIn profile")

            # Chat history display
            chatbot = gr.Chatbot(height=500)

            # User input
            chat_input = gr.Textbox(
                label="Your Question",
                placeholder="What is this person's current role?",
            )

            # Send button
            chat_btn = gr.Button("Send")

            # Button click handler
            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

            # Enter key handler
            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

    return demo


if __name__ == "__main__":

    demo = create_gradio_interface()

    # App launch 
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,          
        share=True,               
    )





