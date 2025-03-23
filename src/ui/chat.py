"""
Chat interface components using Streamlit.
"""

import streamlit as st

from src.llm.gemini import GeminiClient


def init_chat_state():
    """Initialize chat-related session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = GeminiClient()


def display_chat_history():
    """Display the chat history on the Streamlit UI."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Process user input and get response from Gemini."""
    if prompt := st.chat_input("Describe your dataset schema..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the updated chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Generating schema..."):
                response = st.session_state.gemini_client.send_message(prompt)
                message_placeholder.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def render_chat_interface():
    """Render the complete chat interface."""
    st.title("Psilo - Dataset Schema Builder")

    st.markdown("""
    ## Build Your Synthetic Dataset Schema
    
    Describe the dataset you want to create and I'll help you define its structure for generating realistic dummy data.
    
    **How to use:**
    1. Describe your desired dataset in natural language
    2. I'll suggest a schema with appropriate column types and generation methods
    3. You can refine the schema through our conversation
    4. When satisfied, we'll set up data generation parameters
    
    **Examples:**
    - "I need a customer database with names, contact info, purchase history, and loyalty status"
    - "Create a medical patient records schema with demographics and visit history"
    - "Help me design an employee dataset for an HR system with departments and salary information"
    """)

    # Initialize chat state
    init_chat_state()

    # Display chat history
    display_chat_history()

    # Handle user input
    handle_user_input()

    # Add a reset button
    if st.button("Reset Schema Builder"):
        st.session_state.messages = []
        st.session_state.gemini_client.reset_chat()
        st.rerun()
