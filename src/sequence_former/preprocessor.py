import os
import logging
from typing import List, Dict, Any
from .utils import read_file_content

def _enrich_page_content(page_content: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Enriches each line in a string of page content with page and line numbers.

    Args:
        page_content: The string content of a single page/chunk.
        page_number: The page number to assign to the lines.

    Returns:
        A list of dictionaries, where each dictionary represents an enriched line.
        e.g., [{'page': 1, 'line': 1, 'text': 'Hello world'}]
    """
    enriched_lines = []
    lines = page_content.splitlines()
    for i, line_text in enumerate(lines):
        if line_text.strip(): # Only include non-empty lines
            enriched_lines.append({
                "page": page_number,
                "line": i + 1,
                "text": line_text
            })
    return enriched_lines

def _presplit_content(content: str, chunk_size: int) -> List[str]:
    """
    Internal function to split a long string content into several large chunks.
    """
    chunks = []
    start = 0
    while start < len(content):
        if start + chunk_size >= len(content):
            end = len(content)
        else:
            end = content.rfind('\n', start, start + chunk_size)
            if end == -1 or end <= start:
                end = start + chunk_size
        
        chunk_text = content[start:end]
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        # Move start past the current end and skip any leading whitespace
        start = end
        while start < len(content) and content[start].isspace():
            start += 1
        
    return chunks

def load_and_enrich_document(path: str, long_doc_chunk_size: int = 8000) -> List[List[Dict[str, Any]]]:
    """
    Loads a document, pre-splits it into pages/chunks, and enriches each line
    with metadata (page and line numbers).

    Args:
        path: Absolute path to a file (long doc) or a directory (paginated doc).
        long_doc_chunk_size: The character size for splitting long documents.

    Returns:
        A list of pages. Each page is a list of enriched line dictionaries.
        e.g., [ [{'page':1, 'line':1, 'text':'...'}] , [{'page':2, ...}] ]
    """
    if os.path.isdir(path):
        # Filter for markdown files only
        file_paths = sorted([
            os.path.join(path, f) 
            for f in os.listdir(path) 
            if os.path.isfile(os.path.join(path, f)) and f.endswith('.md')
        ])
        
        if not file_paths:
            logging.warning(f"No markdown files (.md) found in directory: {path}")
            return []

        pages_content = []
        for p in file_paths:
            content = read_file_content(p)
            if not content.startswith("Error: Could not decode file"):
                pages_content.append(content)
            else:
                logging.warning(f"Skipping unreadable file: {os.path.basename(p)}")

    elif os.path.isfile(path):
        content = read_file_content(path)
        pages_content = _presplit_content(content, chunk_size=long_doc_chunk_size)
    else:
        raise ValueError(f"Path is not a valid file or directory: {path}")

    enriched_document = []
    for i, page_content in enumerate(pages_content):
        page_number = i + 1
        enriched_page = _enrich_page_content(page_content, page_number)
        if enriched_page: # Only add pages that have content
            enriched_document.append(enriched_page)
            
    return enriched_document