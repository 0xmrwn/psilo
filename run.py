#!/usr/bin/env python
"""
Alternative entry point for the Psilo application.
Run with: python run.py
"""

import streamlit.web.cli as stcli
import os
import sys

if __name__ == "__main__":
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Path to the Streamlit app script
    app_path = os.path.join(dir_path, "src", "main.py")

    # Run the Streamlit app
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    sys.exit(stcli.main())
