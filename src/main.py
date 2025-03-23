"""
Main entry point for the Psilo application.
"""

import streamlit as st
from src.ui.chat import render_chat_interface


def main():
    """Main function to run the Psilo application."""
    # Set page configuration
    st.set_page_config(
        page_title="Psilo - Dataset Generator",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Render the chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
