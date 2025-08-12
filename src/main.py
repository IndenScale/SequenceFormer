# src/main.py
import argparse
import logging
import sys

from .client import TextFormerClient
from .config import load_settings

def setup_logging(debug: bool):
    """Configures the logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    log_file = "debug.log" if debug else "textformer.log" # Updated log file name

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )

    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging initialized. Level: {logging.getLevelName(level)}. Log file: {log_file}")

def main():
    parser = argparse.ArgumentParser(description="TextFormer: An intelligent, multi-modal document structuring engine.", add_help=False)

    # --- Argument Groups ---
    required_args = parser.add_argument_group('Required arguments')
    conn_args = parser.add_argument_group('Connection arguments')
    proc_args = parser.add_argument_group('Processing arguments')
    output_args = parser.add_argument_group('Output arguments')

    # --- Argument Definitions ---
    required_args.add_argument("input_path", type=str, help="Absolute path to the input document or directory.")

    conn_args.add_argument("--openai_api_key", type=str, help="OpenAI API Key for text processing.")
    conn_args.add_argument("--openai_base_url", type=str, help="OpenAI-compatible base URL for text.")
    conn_args.add_argument("--openai_model", type=str, help="Text model name.")
    conn_args.add_argument("--openai_vlm_api_key", type=str, help="OpenAI API Key for vision processing.")
    conn_args.add_argument("--openai_vlm_base_url", type=str, help="OpenAI-compatible base URL for vision.")
    conn_args.add_argument("--openai_vlm_model", type=str, help="Vision model name.")

    proc_args.add_argument("--long_doc_chunk_size", type=int, help="Character size for pre-splitting long docs.")
    proc_args.add_argument("--target_chunk_size", type=int, help="Ideal character size for final chunks.")
    proc_args.add_argument("--chunk_size_tolerance", type=float, help="Tolerance for chunk size deviation.")
    proc_args.add_argument("--metadata-schema", type=str, dest="metadata_schema_path", help="Path to the JSON file for the metadata schema.")

    output_args.add_argument("-o", "--output", type=str, help="Path for the output JSONL file. Use '-' for stdout.")
    output_args.add_argument("--debug", action="store_true", help="Enable debug logging.")
    output_args.add_argument("--retain-intermediate", action="store_true", help="Retain intermediate files for debugging.")
    output_args.add_argument("--enable-vlm", action="store_true", help="Enable VLM processing for embedded images.")
    output_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')

    # --- Execution ---
    args = parser.parse_args()

    settings = load_settings(vars(args))
    setup_logging(settings.debug)

    try:
        client = TextFormerClient(settings)
        client.process_file(settings.input_path, settings.output)
    except Exception as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()