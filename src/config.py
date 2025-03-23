import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Default configurations
DEFAULT_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.7
