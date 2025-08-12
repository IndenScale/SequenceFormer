import logging
import os
import re
from typing import Optional

"""
This file contains utility functions that can be shared across different modules.
"""

def read_file_content(file_path: str) -> str:
    """
    Reads the content of a file, trying UTF-8 first and then falling back to GBK.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        logging.warning(f"Failed to read {os.path.basename(file_path)} with UTF-8. Falling back to GBK.")
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read {os.path.basename(file_path)} with both UTF-8 and GBK. Error: {e}")
            # Return an empty string or raise the exception, depending on desired behavior.
            # For robustness, we'll return a string indicating the failure.
            return f"Error: Could not decode file {os.path.basename(file_path)}."


def find_long_fenced_block(text: str, threshold: int) -> bool:
    """
    Finds if there is a fenced code block longer than a given threshold in the text.

    Args:
        text: The text to search within.
        threshold: The length threshold for the content of the fenced block.

    Returns:
        True if a long fenced block is found, False otherwise.
    """
    # Regex to find all fenced blocks (non-greedy)
    pattern = re.compile(r"```[a-z]*\n(.*?)\n```", re.DOTALL)
    for match in pattern.finditer(text):
        # Check the length of the content inside the backticks
        if len(match.group(1)) > threshold:
            return True
    return False