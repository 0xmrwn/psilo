"""
Chat interface components using Streamlit.
"""

import streamlit as st

from src.llm.gemini import GeminiClient, DataSchema, Column


def init_chat_state():
    """Initialize chat-related session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = GeminiClient()

    if "schema" not in st.session_state:
        st.session_state.schema = None

    if "schema_generated" not in st.session_state:
        st.session_state.schema_generated = False


def display_chat_history():
    """Display the chat history on the Streamlit UI."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_schema(schema: DataSchema):
    """Display the generated schema in a nice format."""
    with st.container(border=True):
        st.markdown(f"## {schema.table_name}")
        st.markdown(f"*{schema.description}*")

        # Create a table for the columns
        data = [[col.name, col.type, col.description] for col in schema.columns]

        st.dataframe(
            data=data,
            column_config={
                0: st.column_config.TextColumn("Column Name"),
                1: st.column_config.TextColumn("Data Type"),
                2: st.column_config.TextColumn("Description"),
            },
            hide_index=True,
        )


def handle_user_input():
    """Process user input and get response from Gemini."""
    if prompt := st.chat_input(
        "Describe your dataset schema or refine the current one..."
    ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the updated chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if this is a refinement or a new schema request
        if st.session_state.schema_generated:
            # This is a refinement request
            with st.chat_message("assistant"):
                try:
                    # Use the same two-step process for refinements as for initial schema creation
                    st.markdown("### Thinking through your refinement request...")
                    thinking_placeholder = st.empty()

                    # Create a context message that includes the current schema details
                    current_schema = st.session_state.schema
                    context_message = f"""Refinement request: {prompt}
                    
                    Current schema:
                    Table: {current_schema.table_name}
                    Description: {current_schema.description}
                    Columns:
                    {", ".join([f"{col.name} ({col.type}): {col.description}" for col in current_schema.columns])}
                    """

                    # Stream the thinking process
                    thinking_output = ""
                    with st.spinner("Analyzing your refinement request..."):
                        for chunk in st.session_state.gemini_client.generate_thinking(
                            context_message
                        ):
                            thinking_output += chunk
                            thinking_placeholder.markdown(thinking_output)

                    # Second step: Generate revised structured schema
                    st.markdown("### Updating schema based on your refinement...")
                    with st.spinner("Creating updated schema..."):
                        updated_schema = st.session_state.gemini_client.generate_schema(
                            context_message, thinking_output
                        )

                        # Check if it's an error schema
                        if updated_schema.table_name == "Error Schema":
                            st.warning("⚠️ " + updated_schema.description)
                            st.info(
                                "I'll continue with the existing schema but note your refinement request."
                            )

                            # Add error info to chat history
                            error_message = f"""
                            **API Rate Limit Reached**
                            
                            I encountered a rate limit error while trying to update your schema.
                            
                            Please try again in a few minutes or simplify your refinement request.
                            """

                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_message}
                            )
                        else:
                            # Update the session state with the new schema
                            st.session_state.schema = updated_schema

                            # Display the updated schema
                            st.markdown(
                                "### Here's the updated schema based on your refinement:"
                            )
                            display_schema(updated_schema)

                            # Add the formatted response to chat history
                            refinement_response = f"""
                            I've updated the schema based on your refinement request.
                            
                            The updated schema for **{updated_schema.table_name}** now has {len(updated_schema.columns)} columns.
                            You can see the detailed schema with column names, types, and descriptions above.
                            
                            **You can continue to:**
                            - Ask questions about the schema
                            - Request additional changes
                            - Provide more context for further refinements
                            """

                            st.session_state.messages.append(
                                {"role": "assistant", "content": refinement_response}
                            )
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    st.info(
                        "This might be due to API rate limits. Please try again in a few moments."
                    )
        else:
            # This is a new schema request - use the two-step process
            with st.chat_message("assistant"):
                try:
                    st.markdown("### Thinking through your dataset requirements...")
                    thinking_placeholder = st.empty()

                    # Stream the thinking process
                    thinking_output = ""
                    with st.spinner("Analyzing your requirements..."):
                        for chunk in st.session_state.gemini_client.generate_thinking(
                            prompt
                        ):
                            thinking_output += chunk
                            thinking_placeholder.markdown(thinking_output)

                    # Second step: Generate structured schema
                    st.markdown("### Generating schema based on analysis...")
                    with st.spinner("Creating structured schema..."):
                        schema = st.session_state.gemini_client.generate_schema(
                            prompt, thinking_output
                        )

                        # Check if it's an error schema
                        if schema.table_name == "Error Schema":
                            st.warning("⚠️ " + schema.description)
                            st.session_state.schema = schema
                            st.session_state.schema_generated = True
                            display_schema(schema)

                            # Add error info to chat history
                            error_message = f"""
                            **API Rate Limit Reached**
                            
                            I encountered a rate limit error while trying to generate your schema.
                            
                            The schema displayed might be incomplete. You could:
                            1. Wait a few minutes and try again
                            2. Simplify your dataset request
                            3. Continue with this basic schema and refine it
                            """

                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_message}
                            )
                        else:
                            st.session_state.schema = schema
                            st.session_state.schema_generated = True

                            # Display a summary of the thinking
                            st.markdown(
                                "### Here's the schema I've created based on your requirements:"
                            )

                            # Display the structured schema
                            display_schema(schema)

                            # Add the formatted response to chat history
                            schema_description = f"""
                            I've analyzed your requirements and created a schema for **{schema.table_name}**.
                            
                            This dataset contains {len(schema.columns)} columns covering the essential data elements needed.
                            You can see the detailed schema with column names, types, and descriptions above.
                            
                            **You can now:**
                            - Ask questions about the schema
                            - Request specific changes (add columns, modify data types, etc.)
                            - Provide additional context to refine the schema
                            """

                            st.session_state.messages.append(
                                {"role": "assistant", "content": schema_description}
                            )
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    st.info(
                        "This might be due to API rate limits. Please try again in a few moments."
                    )

                    # Create a minimal error schema
                    error_schema = DataSchema(
                        table_name="Error Schema",
                        description="Failed to generate schema due to API error",
                        columns=[
                            Column(
                                name="error",
                                type="string",
                                description="Schema generation failed. Please try again later.",
                            )
                        ],
                    )

                    st.session_state.schema = error_schema
                    st.session_state.schema_generated = True


def render_chat_interface():
    """Render the complete chat interface."""
    st.title("Psilo - Dataset Schema Builder")

    st.markdown("""
    ## Build Your Synthetic Dataset Schema
    
    Describe the dataset you want to create and I'll help you define its structure for generating realistic dummy data.
    
    **How it works:**
    1. Describe your desired dataset in natural language
    2. I'll think through and analyze your requirements
    3. Then I'll create a structured schema with appropriate column types
    4. You can refine the schema through our conversation
    
    **Examples:**
    - "I need a customer database with names, contact info, purchase history, and loyalty status"
    - "Create a medical patient records schema with demographics and visit history"
    - "Help me design an employee dataset for an HR system with departments and salary information"
    """)

    # Initialize chat state
    init_chat_state()

    # Display chat history
    display_chat_history()

    # Display current schema if it exists
    if st.session_state.schema is not None:
        with st.expander("Current Schema", expanded=True):
            display_schema(st.session_state.schema)

            # Add schema refinement guidance
            st.markdown("""
            ### Refine Your Schema
            
            You can refine this schema by:
            - Adding new columns: "Please add a column for customer age"
            - Modifying data types: "Change the email column to string type"
            - Removing columns: "Remove the middle_name column"
            - Requesting explanations: "Why did you choose these columns?"
            """)

    # Handle user input
    handle_user_input()

    # Add a reset button
    if st.button("Reset Schema Builder"):
        st.session_state.messages = []
        st.session_state.schema = None
        st.session_state.schema_generated = False
        st.session_state.gemini_client.reset_chat()
        st.rerun()
