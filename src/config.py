import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Default configurations
DEFAULT_MODEL = "gemini-1.5-pro"
TEMPERATURE = 0.7
