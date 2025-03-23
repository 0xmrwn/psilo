# Psilo

Psilo is an intelligent dummy data generator that uses natural language processing to help you define dataset schemas and generate synthetic data.

## Features

- Chat-based schema definition interface
- Uses Gemini AI to understand and interpret your data requirements
- Integration with Faker for common data types
- Interactive UI for previewing and customizing data generation
- Export data to CSV and other formats

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -e .
   ```
3. Create a `.env` file in the root directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### Running the Application

You can run the application in one of the following ways:

**Option 1:** Using Python
```
python main.py
```

**Option 2:** Directly with Streamlit
```
streamlit run src/main.py
```

**Option 3:** Using the run script
```
python run.py
```

### How to Use

1. When the application starts, you'll see a chat interface
2. Describe the dataset you want to generate using natural language
3. The AI will interpret your requirements and suggest a schema
4. Once you're satisfied with the schema, you can customize generation parameters
5. Finally, generate your dataset and export it in your preferred format

## Project Structure

- `src/`: Main source code
  - `llm/`: Integration with language models (Gemini)
  - `ui/`: Streamlit UI components
  - `data/`: Data generation logic
  - `persistence/`: File handling and export functionality
  - `utils/`: Utility functions
- `tests/`: Test cases

## License

This project is licensed under the terms of the license included in the repository.
