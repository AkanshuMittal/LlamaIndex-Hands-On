import streamlit as st
import uuid
import logging

from modules.data_extraction import extract_mock_profile, extract_linkedin_profile
from modules.data_processing import (
    split_profile_data,
    create_vector_database,
    verify_embeddings,
)
from modules.query_engine import (
    generate_initial_facts,
    answer_user_query,
)

from modules.pdf_extraction import extract_profile_from_pdf
import config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Streamlit Page Config
st.set_page_config(
    page_title="LinkedIn Icebreaker Bot",
    layout="wide"
)

st.title("🔗 LinkedIn Icebreaker Bot")
st.write(
    "Generate personalized icebreakers and ask questions using a "
    "**Retrieval-Augmented Generation (RAG)** pipeline."
)


# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

# 🔹 PROFILE PROCESSING SECTION
st.header("1️⃣ Process LinkedIn Profile")

source_type = st.radio(
    "Select Profile Source",
    ["PDF Upload", "LinkedIn URL", "JSON (Mock Data)"]
)

with st.form("profile_form"):
    linkedin_url = None
    uploaded_pdf = None
        
    if source_type == "LinkedIn URL":
        linkedin_url = st.text_input(
            "Enter LinkedIn Profile URL",
            placeholder="https://www.linkedin.com/in/username/"
        )
    
    elif source_type == "PDF Upload":
        uploaded_pdf = st.file_uploader(
            "Upload LinkedIn Profile PDF",
            type=["pdf"]
        )
        
    submit_btn = st.form_submit_button("Process Profile")

if submit_btn:
    try:
        if source_type == "PDF Upload":
            if not uploaded_pdf:
                st.warning("⚠️ Please upload a PDF file.")
                st.stop()
                
            profile_data = extract_profile_from_pdf(uploaded_pdf)
            
        elif source_type == "LinkedIn URL":
            if not linkedin_url:
                st.warning("⚠️ Please enter a LinkedIn profile URL.")
                st.stop()
                
            profile_data = extract_linkedin_profile(linkedin_url)
            
        elif source_type == "JSON (Mock Data)":
            profile_data = extract_mock_profile()
            

        with st.spinner("🧩 Splitting profile data into chunks..."):
            nodes = split_profile_data(profile_data)

        if not nodes:
            st.error("❌ Failed to process profile data.")
            st.stop()

        with st.spinner("📦 Creating vector database..."):
            index = create_vector_database(nodes)

        if not index:
            st.error("❌ Failed to create vector database.")
            st.stop()

        verify_embeddings(index)

        with st.spinner("✨ Generating initial facts..."):
            facts = generate_initial_facts(index)

        # Save session data
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.vector_index = index

        st.success("✅ Profile processed successfully!")
        st.subheader("📌 3 Interesting Facts")
        st.write(facts)

    except Exception as e:
        logger.error(e)
        st.error(f"❌ Error: {e}")

# 🔹 CHAT SECTION
st.header("2️⃣ Chat with Profile")

if st.session_state.vector_index is None:
    st.info("ℹ️ Please process a LinkedIn profile first.")
else:
    user_query = st.text_input(
        "Ask a question about this profile",
        placeholder="What is this person's current role?"
    )

    if st.button("Ask"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            try:
                with st.spinner("🤖 Thinking..."):
                    response = answer_user_query(
                        st.session_state.vector_index,
                        user_query
                    )

                st.subheader("💬 Answer")
                st.write(response.response)

            except Exception as e:
                logger.error(e)
                st.error(f"❌ Error: {e}")
