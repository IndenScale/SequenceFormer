import os
import logging
import json
import glob
from typing import List, Dict, Any, Optional

from .utils import read_file_content

def _enrich_page_content(page_content: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Enriches each line in a string of page content with page and line numbers.
    """
    enriched_lines = []
    lines = page_content.splitlines()
    for i, line_text in enumerate(lines):
        if line_text.strip():
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
        end = content.rfind('\n', start, start + chunk_size)
        if end == -1 or end <= start:
            end = start + chunk_size
        chunks.append(content[start:end])
        start = end + 1
    return chunks

def _load_mineru_document(path: str) -> Optional[List[str]]:
    """
    Loads a document from a MinerU output directory, prioritizing content_list.json.
    """
    content_list_path_glob = os.path.join(path, '*_content_list.json')
    content_list_paths = glob.glob(content_list_path_glob)
    
    if not content_list_paths:
        logging.warning(f"MinerU mode enabled, but no '*_content_list.json' found in {path}.")
        return None
        
    content_list_path = content_list_paths[0]
    logging.info(f"Found MinerU content list: {content_list_path}")

    try:
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read or parse {content_list_path}: {e}")
        return None

    pages: Dict[int, List[str]] = {}
    for item in content_list:
        page_idx = item.get('page_idx', 0)
        if page_idx not in pages:
            pages[page_idx] = []
        
        item_type = item.get('type')
        text_content = ""

        if item_type == 'text':
            text_content = item.get('text', '')
        elif item_type == 'image':
            caption = "".join(item.get('image_caption', []))
            img_path = item.get('img_path', '')
            if img_path:
                # Convert to markdown format recognizable by postprocessor
                text_content = f"![{caption}]({img_path})"
        elif item_type == 'table':
            caption = "".join(item.get('table_caption', []))
            table_body = item.get('table_body', '')
            if caption:
                text_content = f"{caption}\n{table_body}"
            else:
                text_content = table_body
        elif item_type == 'equation':
            text_content = item.get('text', '')

        if text_content:
            pages[page_idx].append(text_content)

    sorted_pages = sorted(pages.items())
    
    # Each page's content is concatenated into a single string.
    # This treats each page as a "page" for the downstream processor.
    pages_content = ["\n".join(content) for _, content in sorted_pages]
    
    return pages_content

def load_and_enrich_document(
    path: str, 
    long_doc_chunk_size: int = 8000,
    mineru_enabled: bool = False
) -> List[List[Dict[str, Any]]]:
    """
    Loads a document, pre-splits it, and enriches each line with metadata.
    Supports standard files, directories of markdown, and MinerU directories.
    """
    pages_content: Optional[List[str]] = None

    if os.path.isdir(path):
        if mineru_enabled:
            pages_content = _load_mineru_document(path)
        
        # Fallback to reading markdown files if MinerU processing fails or is disabled
        if pages_content is None:
            logging.info("Processing directory as a collection of markdown files.")
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
        logging.info("Processing as a single long document file.")
        content = read_file_content(path)
        pages_content = _presplit_content(content, chunk_size=long_doc_chunk_size)
    else:
        raise ValueError(f"Path is not a valid file or directory: {path}")

    if pages_content is None:
        return []

    enriched_document = []
    for i, page_content in enumerate(pages_content):
        page_number = i + 1
        enriched_page = _enrich_page_content(page_content, page_number)
        if enriched_page:
            enriched_document.append(enriched_page)
            
    return enriched_document

def prepare_llm_input(
    unprocessed_lines: List[Dict[str, Any]],
    target_chunk_size: int,
    chunk_size_tolerance: float
) -> str:
    """
    Prepares the text input for the LLM by selecting a batch of unprocessed lines.
    """
    max_chars = target_chunk_size * (1 + chunk_size_tolerance)
    current_chars = 0
    lines_for_batch = []
    
    for line in unprocessed_lines:
        line_text = line['text']
        line_len = len(line_text)
        
        if current_chars + line_len + 1 > max_chars and lines_for_batch:
            break
        
        lines_for_batch.append(f"p{line['page']}:l{line['line']}:{line_text}")
        current_chars += line_len + 1
        
    return "\n".join(lines_for_batch)
