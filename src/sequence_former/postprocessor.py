import logging
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from .data_models import ProcessingState, LLMOutput, Chunk
from .config import Settings
from .utils import find_long_fenced_block
from .llm_processor import call_vlm

# ... (existing functions _validate_metadata, _validate_chunk_size, etc. remain the same)
def _validate_metadata(chunk: Chunk, settings: Settings):
    """(Placeholder) Validates a chunk's metadata against the schema."""
    pass

def _validate_chunk_size(chunk: Chunk, settings: Settings):
    """Validates chunk size, skipping for chunks with large fenced content."""
    if chunk.raw_text is None: return
    max_len = settings.target_chunk_size * (1 + settings.chunk_size_tolerance)
    if len(chunk.raw_text) > max_len:
        fenced_threshold = int(settings.target_chunk_size * (1 - settings.chunk_size_tolerance))
        if find_long_fenced_block(chunk.raw_text, fenced_threshold):
            logging.debug(f"Chunk size validation skipped for chunk at p{chunk.start_page}:l{chunk.start_line} due to large fenced block.")
            return
        logging.warning(f"Chunk at p{chunk.start_page}:l{chunk.start_line} exceeds max size. Length: {len(chunk.raw_text)}, Max: {int(max_len)}.")

def _get_line_identifier(line: Dict[str, Any]) -> str:
    """Creates a unique string identifier for a line."""
    return f"p{line['page']}:l{line['line']}"

def _reconstruct_chunk_text(chunk: Chunk, all_lines_map: Dict[str, str]) -> str:
    """Reconstructs the raw text of a chunk, handling multi-page chunks."""
    text_lines = []
    for page_num in range(chunk.start_page, chunk.end_page + 1):
        start_line = chunk.start_line if page_num == chunk.start_page else 1
        end_line = chunk.end_line if page_num == chunk.end_page else 9999
        for line_num in range(start_line, end_line + 1):
            key = f"p{page_num}:l{line_num}"
            if key in all_lines_map:
                text_lines.append(all_lines_map[key])
    return "\n".join(text_lines)

def populate_chunk_text(chunk: Chunk, all_lines_map: Dict[str, str]) -> None:
    """A utility function to reconstruct and populate the raw_text field of a chunk."""
    chunk.raw_text = _reconstruct_chunk_text(chunk, all_lines_map)

def _associate_chunk_with_heading(chunk: Chunk, hierarchical_headings: List[str]) -> None:
    """
    为chunk自动关联最合适的heading。
    
    规则：
    1. 如果chunk已经有heading（由LLM返回），则保持不变
    2. 如果chunk没有heading，则关联到最后一个有效的heading
    3. 如果没有可用的heading，则保持为None
    """
    if chunk.heading is not None and chunk.heading.strip():
        # LLM已经返回了heading，保持不变
        return
    
    if hierarchical_headings:
        # 自动关联到最后一个heading
        chunk.heading = hierarchical_headings[-1]
    else:
        # 没有可用的heading，保持为None
        chunk.heading = None

def _merge_small_chunks(chunks: List[Chunk], settings: Settings) -> List[Chunk]:
    """Merges chunks that are smaller than the minimum size with adjacent chunks under the same heading."""
    if not chunks or settings.min_chunk_size <= 0:
        return chunks

    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        
        if chunk.raw_text and len(chunk.raw_text) < settings.min_chunk_size:
            # Prioritize merging with the previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                if prev_chunk.heading == chunk.heading:
                    logging.debug(f"Merging small chunk (p{chunk.start_page}:l{chunk.start_line}) into previous chunk (p{prev_chunk.start_page}:l{prev_chunk.start_line}).")
                    prev_chunk.raw_text += "\n" + chunk.raw_text
                    prev_chunk.end_page = chunk.end_page
                    prev_chunk.end_line = chunk.end_line
                    prev_chunk.summary += " " + chunk.summary
                    chunks.pop(i)
                    continue
            
            # If not merged, try merging with the next chunk
            if i + 1 < len(chunks):
                next_chunk = chunks[i+1]
                if next_chunk.heading == chunk.heading:
                    logging.debug(f"Merging small chunk (p{chunk.start_page}:l{chunk.start_line}) into next chunk (p{next_chunk.start_page}:l{next_chunk.start_line}).")
                    next_chunk.raw_text = chunk.raw_text + "\n" + next_chunk.raw_text
                    next_chunk.start_page = chunk.start_page
                    next_chunk.start_line = chunk.start_line
                    next_chunk.summary = chunk.summary + " " + next_chunk.summary
                    chunks.pop(i)
                    continue

        i += 1
        
    return chunks

def finalize_chunks_and_update_state(
    llm_output: LLMOutput,
    combined_enriched_lines: List[Dict[str, Any]],
    current_page_boundary: int,
    current_state: ProcessingState,
    settings: Settings
) -> Tuple[List[Chunk], ProcessingState]:
    """Processes the LLM output to finalize reliable chunks and update the state."""
    all_lines_map = {_get_line_identifier(line): line['text'] for line in combined_enriched_lines}
    
    # 1. Populate raw_text and associate headings for all chunks from the LLM output first.
    for chunk in llm_output.chunks:
        populate_chunk_text(chunk, all_lines_map)
        _associate_chunk_with_heading(chunk, llm_output.hierarchical_headings)

    # 2. Merge small chunks if the setting is enabled.
    processed_chunks = _merge_small_chunks(llm_output.chunks, settings)

    processed_line_ids = set(current_state.processed_lines)

    consumed_line_ids = set()
    for chunk in processed_chunks:
        for page_num in range(chunk.start_page, chunk.end_page + 1):
            start_line = chunk.start_line if page_num == chunk.start_page else 1
            end_line = chunk.end_line if page_num == chunk.end_page else 9999
            for line_num in range(start_line, end_line + 1):
                line_id = f"p{page_num}:l{line_num}"
                if line_id in all_lines_map:
                    consumed_line_ids.add(line_id)

    unprocessed_lines = [
        line for line in combined_enriched_lines 
        if _get_line_identifier(line) not in consumed_line_ids
    ]

    reliable_chunks: List[Chunk] = []
    new_staged_chunk: Optional[Chunk] = None
    
    if processed_chunks:
        new_staged_chunk = processed_chunks[-1]
        reliable_chunks = processed_chunks[:-1]

    for chunk in reliable_chunks:
        _validate_chunk_size(chunk, settings)
        _validate_metadata(chunk, settings)
        for page_num in range(chunk.start_page, chunk.end_page + 1):
            start_line = chunk.start_line if page_num == chunk.start_page else 1
            end_line = chunk.end_line if page_num == chunk.end_page else 9999
            for line_num in range(start_line, end_line + 1):
                line_id = f"p{page_num}:l{line_num}"
                if line_id in all_lines_map:
                    processed_line_ids.add(line_id)

    new_state = ProcessingState(
        doc_id=current_state.doc_id,
        hierarchical_headings=llm_output.hierarchical_headings,
        metadata_schema=current_state.metadata_schema,
        staged_chunk=new_staged_chunk,
        unprocessed_lines=unprocessed_lines,
        processed_lines=processed_line_ids
    )
    return reliable_chunks, new_state

# --- VLM Enrichment (Refactored) ---

def _create_text_chunk_from_slice(
    text_slice: str, 
    original_chunk: Chunk, 
    line_offset: int, 
    total_lines_in_slice: int
) -> Chunk:
    """Helper to create a new text chunk from a slice of the original."""
    new_chunk = original_chunk.model_copy(deep=True)
    new_chunk.raw_text = text_slice
    # Approximate the new summary
    new_chunk.summary = f"(Text segment from original chunk) {original_chunk.summary}"
    # Approximate the new line numbers
    new_chunk.start_line += line_offset
    new_chunk.end_line = new_chunk.start_line + total_lines_in_slice - 1
    return new_chunk

def enrich_chunks_with_vlm(chunks: List[Chunk], settings: Settings) -> List[Chunk]:
    """
    Deconstructs and reconstructs chunks containing images, creating dedicated image chunks.
    Returns a new list of chunks.
    """
    logging.info("Starting VLM post-enrichment process (Deconstruction/Reconstruction mode)...")
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    input_dir = os.path.dirname(settings.input_path) if os.path.isfile(settings.input_path) else settings.input_path
    
    reconstructed_chunks: List[Chunk] = []
    TEXT_SLICE_THRESHOLD = 250

    for chunk in chunks:
        if not chunk.raw_text or not image_pattern.search(chunk.raw_text):
            reconstructed_chunks.append(chunk)
            continue

        logging.info(f"Found image reference(s) in chunk p{chunk.start_page}:l{chunk.start_line}. Deconstructing...")
        
        last_end = 0
        text_slices = []
        image_infos = []

        for match in image_pattern.finditer(chunk.raw_text):
            text_before = chunk.raw_text[last_end:match.start()]
            text_slices.append(text_before)
            image_infos.append({
                "tag": match.group(0),
                "alt": match.group(1),
                "path": match.group(2)
            })
            last_end = match.end()
        text_slices.append(chunk.raw_text[last_end:])

        current_line_offset = 0
        for i, text_slice in enumerate(text_slices):
            slice_line_count = text_slice.count('\n')
            
            if len(text_slice.strip()) > TEXT_SLICE_THRESHOLD:
                new_text_chunk = _create_text_chunk_from_slice(text_slice, chunk, current_line_offset, slice_line_count)
                reconstructed_chunks.append(new_text_chunk)
            elif i > 0 and reconstructed_chunks: # Merge small slice with previous chunk
                reconstructed_chunks[-1].raw_text += text_slice
                reconstructed_chunks[-1].end_line += slice_line_count
            
            current_line_offset += slice_line_count

            if i < len(image_infos): # If there's an image to process after this slice
                info = image_infos[i]
                image_absolute_path = os.path.abspath(os.path.join(input_dir, info['path']))
                
                if not os.path.exists(image_absolute_path):
                    logging.warning(f"Image file not found: {image_absolute_path}. Skipping.")
                    continue
                
                try:
                    vlm_description = call_vlm(image_absolute_path, chunk.raw_text, settings)
                    enriched_block = f"""```image\npath: \"{info['path']}\"\ncaption: \"{info['alt']}\"\ndescription: \"{vlm_description}\"\n```"""
                    
                    image_chunk = chunk.model_copy(deep=True)
                    image_chunk.raw_text = enriched_block
                    image_chunk.summary = f"Image: {info['alt'] or 'Untitled'} - {vlm_description[:100]}..."
                    image_chunk.metadata = {"type": "Figure", "form": "figure"}
                    image_chunk.start_line += current_line_offset
                    image_chunk.end_line = image_chunk.start_line # Image chunk is a single "line"
                    reconstructed_chunks.append(image_chunk)
                    current_line_offset += enriched_block.count('\n')

                except Exception as e:
                    logging.error(f"Failed to process image '{info['path']}' with VLM: {e}")

    logging.info("VLM post-enrichment finished.")
    return reconstructed_chunks