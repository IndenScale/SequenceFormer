import json
import logging
import os
import sys
from typing import List, Dict, Any

from .data_models import ProcessingState, Chunk
from .preprocessor import load_and_enrich_document
from .llm_processor import process_page_with_llm
from .postprocessor import finalize_chunks_and_update_state, populate_chunk_text, enrich_chunks_with_vlm
from .config import Settings

class SequenceFormerClient:
    """
    A client for the SequenceFormer intelligent document processing pipeline.
    """
    def __init__(self, settings: Settings):
        self.settings = settings

    def process_file(self, input_path: str, output_path: str):
        """
        Processes a document from a file path and writes the output to another file or stdout.
        """
        self.settings.input_path = input_path # Ensure input_path is set for VLM

        logging.info(f"Starting processing for: {input_path}")
        enriched_document = load_and_enrich_document(input_path, self.settings.long_doc_chunk_size)

        doc_id = os.path.basename(input_path)
        initial_heading = f"# {doc_id}"

        initial_metadata_schema = self._load_metadata_schema()

        state = ProcessingState(
            doc_id=doc_id,
            hierarchical_headings=[initial_heading],
            metadata_schema=initial_metadata_schema
        )

        output_stream = open(output_path, 'w', encoding='utf-8') if output_path != "-" else sys.stdout
        try:
            for i, current_page_lines in enumerate(enriched_document):
                page_num = i + 1
                logging.info(f"Processing Mega Chunk {page_num}/{len(enriched_document)}...")

                # This is a simplified context management. A full implementation would be more robust.
                combined_input_lines = current_page_lines

                try:
                    llm_output = process_page_with_llm(combined_input_lines, state, self.settings)
                except Exception as e:
                    logging.error(f"Failed to process Mega Chunk {page_num}. Halting. Error: {e}")
                    # In a library, we might re-raise or handle this differently.
                    # For now, we exit for CLI behavior.
                    sys.exit(1)

                page_boundary = current_page_lines[0]['page'] if current_page_lines else 0
                reliable_chunks, state = finalize_chunks_and_update_state(
                    llm_output, combined_input_lines, page_boundary, state, self.settings
                )

                if reliable_chunks:
                    self._process_and_persist_chunks(reliable_chunks, enriched_document, output_stream)

            if state.staged_chunk:
                logging.info("Finalizing the last staged chunk from the document.")
                final_chunk_list = [state.staged_chunk]
                all_lines_map = {f"p{line['page']}:l{line['line']}": line['text'] for page in enriched_document for line in page}
                populate_chunk_text(state.staged_chunk, all_lines_map)

                self._process_and_persist_chunks(final_chunk_list, enriched_document, output_stream, is_final=True)

        finally:
            if output_stream is not sys.stdout:
                output_stream.close()
                logging.info(f"Output written to {output_path}")
            else:
                logging.info("Output written to stdout.")

    def process_text(self, text: str) -> List[Chunk]:
        """
        Processes a string of text in memory and returns a list of Chunk objects.
        Note: VLM processing is disabled in this mode as there's no file context.
        """
        # This method would require a slightly different pre-processing step.
        # For now, it's a placeholder for the library interface.
        raise NotImplementedError("process_text is not yet implemented.")

    def _load_metadata_schema(self) -> Dict[str, Any]:
        """Loads the metadata schema from the path specified in settings."""
        schema = {}
        if self.settings.metadata_schema_path:
            if os.path.exists(self.settings.metadata_schema_path):
                logging.info(f"Loading metadata schema from: {self.settings.metadata_schema_path}")
                with open(self.settings.metadata_schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
            else:
                logging.warning(f"Metadata schema file not found at: {self.settings.metadata_schema_path}. Proceeding without it.")
        else:
            default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'default_metadata_schema.json')
            if os.path.exists(default_path):
                logging.info(f"Loading default metadata schema from: {default_path}")
                with open(default_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
        return schema

    def _process_and_persist_chunks(self, chunks: List[Chunk], enriched_document: List[List[Dict]], stream: Any, is_final: bool = False):
        """Helper to handle VLM enrichment and persistence for a batch of chunks."""
        if self.settings.enable_vlm:
            chunks = enrich_chunks_with_vlm(chunks, self.settings)

        for chunk in chunks:
            stream.write(chunk.model_dump_json() + '\n')

        if is_final:
            logging.info(f"Successfully processed and persisted {len(chunks)} final chunk(s).")
        else:
            logging.info(f"Successfully processed and persisted {len(chunks)} chunks.")
