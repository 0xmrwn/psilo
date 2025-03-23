"""
Main entry point for the Psilo application.
Run with: streamlit run main.py
"""

import subprocess
import sys


def main():
    """
    Run the Streamlit application.
    This allows the app to be launched directly with Python or through Streamlit.
    """
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/main.py"])


if __name__ == "__main__":
    main()
