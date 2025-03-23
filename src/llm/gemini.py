"""
Gemini API integration for handling natural language processing.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Generator, TypeVar
import instructor
from pydantic import BaseModel

from src.config import GEMINI_API_KEY, DEFAULT_MODEL, TEMPERATURE

# Configure the Gemini API with the API key
genai.configure(api_key=GEMINI_API_KEY)

T = TypeVar("T", bound=BaseModel)


class Column(BaseModel):
    """Represents a column in a data schema."""

    name: str
    type: str
    description: str


class DataSchema(BaseModel):
    """Represents a complete data schema."""

    table_name: str
    description: str
    columns: List[Column]


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    # System prompt that defines the context and role of the assistant
    SYSTEM_PROMPT = """
    You are a specialized assistant designed to help users create data schemas for dummy datasets.

    Your primary functions are:
    1. Interpret the user's description of their desired dataset
    2. Generate a comprehensive data schema with appropriate column names and data types
    3. Help users refine their schema through iterative conversation
    4. Make precise updates to existing schemas based on user feedback

    When responding:
    - Format schema suggestions in a clear, structured manner
    - For each column, specify:
      * Column name
      * Data type (string, integer, float, date, boolean, etc.)
      * Description of the data
    - Present your suggestions in markdown tables when appropriate
    - Be helpful in refining the schema based on user feedback
    - Focus exclusively on dataset schema design.
    - When making refinements to an existing schema, ensure you maintain columns and structure that weren't requested to be changed

    Remember that the user will be using this schema to generate synthetic data for testing or development purposes.
    """

    # Thinking prompt to guide the brainstorming process
    THINKING_PROMPT = """
    For this dataset schema design task, I need you to first think through the problem carefully.
    
    If this is a new schema request:
    1. Analyze the user's request to understand what industry or domain they're working in
    2. Identify key entities that would be present in such a dataset
    3. Consider what columns/fields these entities would need
    4. Think about appropriate data types for each field
    5. Consider relationships between different entities if applicable
    
    If this is a refinement to an existing schema:
    1. Carefully analyze the existing schema to understand its structure
    2. Interpret exactly what changes the user is requesting
    3. Determine which columns need to be added, removed, or modified
    4. Consider how the requested changes affect the overall schema
    5. Ensure you preserve existing columns and attributes that weren't requested to be changed
    
    Provide your detailed thinking process as you analyze this request.
    This is just the brainstorming phase - don't create the formal schema yet.
    """

    # Structured output prompt to generate the actual schema
    STRUCTURED_OUTPUT_PROMPT = """
    Now that we've thought through the dataset requirements, create a formal structured schema.
    
    Your output should be a single DataSchema object with:
    1. A descriptive table name
    2. A brief description of the dataset
    3. A list of columns, each with:
       - name: A clear, descriptive column name
       - type: The data type (string, integer, float, date, boolean, etc.)
       - description: A brief explanation of what the column represents
    
    Focus only on the schema design (column names, types, and descriptions).
    Don't include any dummy data generation methods or implementations.
    
    IMPORTANT FOR SCHEMA REFINEMENTS:
    If you are refining an existing schema, ensure that:
    1. Your response includes ALL columns from the original schema unless explicitly requested to remove them
    2. You apply the requested changes accurately (additions, removals, modifications)
    3. The table name and description remain consistent unless changes were requested
    4. You maintain the original attributes of unchanged columns
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
        self.instructor_client = instructor.from_gemini(
            client=self.model_instance, mode=instructor.Mode.GEMINI_JSON
        )
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

    def generate_thinking(self, user_message: str) -> Generator[str, None, None]:
        """
        Generate a thinking/brainstorming process for the given user message.

        Args:
            user_message: The user's description of the desired dataset

        Returns:
            A generator yielding chunks of the thinking process as they become available
        """
        prompt = f"{self.THINKING_PROMPT}\n\nUser request: {user_message}"

        response = self.model_instance.generate_content(prompt, stream=True)

        thinking_output = ""
        for chunk in response:
            if hasattr(chunk, "text"):
                thinking_output += chunk.text
                yield chunk.text
            elif hasattr(chunk, "parts") and chunk.parts:
                thinking_output += chunk.parts[0].text
                yield chunk.parts[0].text

        # Add the thinking process to chat history
        if self.chat_session:
            # Add user message
            self.chat_session.send_message(user_message)
            # Add assistant thinking response
            self.chat_session.send_message(f"[Thinking Process]\n{thinking_output}")

        return thinking_output

    def generate_schema(self, user_message: str, thinking_output: str) -> DataSchema:
        """
        Generate a structured data schema using the Instructor library.

        Args:
            user_message: The original user message
            thinking_output: The thinking/brainstorming output from generate_thinking

        Returns:
            A DataSchema object containing the structured schema
        """
        prompt = f"""
        Original request: {user_message}
        
        Thinking process: {thinking_output}
        
        {self.STRUCTURED_OUTPUT_PROMPT}
        """

        try:
            # Try to generate structured schema using Instructor
            schema = self.instructor_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=DataSchema,
            )

            # Add the schema to the chat session for future context
            schema_summary = f"""
            Based on the analysis, here's the schema I created:
            
            Table: {schema.table_name}
            Description: {schema.description}
            
            Columns:
            {self._format_columns_for_history(schema.columns)}
            """

            # Add the schema summary to the chat history
            if self.chat_session:
                self.chat_session.send_message(schema_summary)

            return schema

        except Exception as e:
            # If structured output fails (rate limit, quota exceeded, etc.), fallback to regular generation
            print(
                f"Structured schema generation failed: {str(e)}. Falling back to regular generation."
            )

            # Fallback to regular generation without Instructor
            fallback_prompt = f"""
            Original request: {user_message}
            
            Thinking process: {thinking_output}
            
            Based on the above, create a data schema with:
            1. A table name
            2. A description
            3. A list of columns with name, type, and description
            
            Format your response so I can parse it into a structured schema.
            Table name should be a simple string.
            Description should be a brief paragraph.
            Columns should be a list where each column has:
            - name: A clear column name (string)
            - type: Data type (string, integer, float, date, boolean, etc.)
            - description: Brief explanation of the column
            """

            try:
                # Use regular chat completion (not structured)
                response = self.model_instance.generate_content(fallback_prompt)
                response_text = response.text

                # Store this in chat history
                if self.chat_session:
                    self.chat_session.send_message(fallback_prompt)
                    self.chat_session.send_message(response_text)

                # Try to parse the response into a schema manually
                try:
                    # Simple parsing attempt - this could be improved
                    lines = response_text.strip().split("\n")
                    table_name = "Default Table"
                    description = "Generated schema"
                    columns = []

                    # Very simple parsing logic
                    current_section = None
                    for line in lines:
                        line = line.strip()
                        if "Table" in line and ":" in line:
                            table_name = line.split(":", 1)[1].strip()
                        elif "Description" in line and ":" in line:
                            description = line.split(":", 1)[1].strip()
                        elif "Columns" in line or "Fields" in line:
                            current_section = "columns"
                        elif (
                            current_section == "columns"
                            and line.startswith("-")
                            and ":" in line
                        ):
                            parts = line.strip("- ").split(":", 1)
                            if len(parts) == 2:
                                name_part = parts[0].strip()
                                desc_part = parts[1].strip()

                                # Try to extract type from name or description
                                type_start = name_part.find("(")
                                type_end = name_part.find(")")
                                if type_start != -1 and type_end != -1:
                                    col_name = name_part[:type_start].strip()
                                    col_type = name_part[
                                        type_start + 1 : type_end
                                    ].strip()
                                else:
                                    col_name = name_part
                                    col_type = "string"  # Default type

                                columns.append(
                                    Column(
                                        name=col_name,
                                        type=col_type,
                                        description=desc_part,
                                    )
                                )

                    return DataSchema(
                        table_name=table_name, description=description, columns=columns
                    )
                except Exception as parse_error:
                    # If parsing fails, create a minimal schema with an error note
                    print(f"Failed to parse fallback response: {str(parse_error)}")
                    return DataSchema(
                        table_name="Error Schema",
                        description="Failed to generate structured schema due to API limits. Please try again later or refine your request.",
                        columns=[
                            Column(
                                name="error",
                                type="string",
                                description="Schema generation failed due to API rate limits. Please check the text response for details.",
                            )
                        ],
                    )
            except Exception as fallback_error:
                # If everything fails, return a minimal error schema
                print(f"Fallback schema generation also failed: {str(fallback_error)}")
                return DataSchema(
                    table_name="Error Schema",
                    description="Failed to generate schema due to API rate limits",
                    columns=[
                        Column(
                            name="error",
                            type="string",
                            description="Schema generation failed due to API rate limits. Please try again later.",
                        )
                    ],
                )

    def _format_columns_for_history(self, columns: List[Column]) -> str:
        """Format columns for the chat history in a readable way."""
        result = ""
        for col in columns:
            result += f"- {col.name} ({col.type}): {col.description}\n"
        return result

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
