"""
Gemini API integration for handling natural language processing.
"""

import google.generativeai as genai
from typing import List, Dict, Any

from src.config import GEMINI_API_KEY, DEFAULT_MODEL, TEMPERATURE

# Configure the Gemini API with the API key
genai.configure(api_key=GEMINI_API_KEY)


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    # System prompt that defines the context and role of the assistant
    SYSTEM_PROMPT = """
    You are a specialized assistant designed to help users create data schemas for dummy datasets.

    Your primary functions are:
    1. Interpret the user's description of their desired dataset
    2. Generate a comprehensive data schema with appropriate column names and data types
    5. Help users refine their schema through iterative conversation

    When responding:
    - Format schema suggestions in a clear, structured manner
    - For each column, specify:
      * Column name
      * Data type (string, integer, float, date, boolean, etc.)
      * Description of the data
    - Present your suggestions in markdown tables when appropriate
    - Be helpful in refining the schema based on user feedback
    - Focus exclusively on dataset schema design.

    Remember that the user will be using this schema to generate synthetic data for testing or development purposes.
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE):
        """
        Initialize the Gemini client.

        Args:
            model: The model name to use (default from config)
            temperature: The temperature setting for generation (default from config)
        """
        self.model = model
        self.temperature = temperature
        self.genai = genai
        self.model_instance = genai.GenerativeModel(
            model_name=model, generation_config={"temperature": temperature}
        )
        self.chat_session = None
        self.reset_chat()

    def reset_chat(self) -> None:
        """Reset the chat session."""
        # Initialize with system prompt as the first message
        self.chat_session = self.model_instance.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "I need your help creating data schemas for dummy datasets."
                    ],
                },
                {"role": "model", "parts": [self.SYSTEM_PROMPT]},
            ]
        )

    def send_message(self, message: str) -> str:
        """
        Send a message to the Gemini API and get a response.

        Args:
            message: The user message to send

        Returns:
            The response from the Gemini API
        """
        if not self.chat_session:
            self.reset_chat()

        response = self.chat_session.send_message(message)
        return response.text

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.

        Returns:
            A list of message dictionaries with 'role' and 'content' keys
        """
        if not self.chat_session:
            return []

        # Skip the first two messages which are the system prompt setup
        history = []
        for i, message in enumerate(self.chat_session.history):
            if i < 2:  # Skip the system prompt messages
                continue
            role = "user" if message.role == "user" else "assistant"
            history.append({"role": role, "content": message.parts[0].text})

        return history
