# SequenceFormer

**SequenceFormer** is an intelligent, multi-modal document structuring engine powered by Large Language Models (LLMs). It transforms unstructured and semi-structured text documents into a stream of clean, semantically coherent, and richly annotated JSON objects called `Atom Chunks`.

This tool is designed to be the robust, foundational layer of any advanced RAG (Retrieval-Augmented Generation) or document intelligence pipeline.

## Core Philosophy

-   **Structure First**: The primary goal is to understand and preserve the structural and formal elements of a document (headings, lists, tables, code, figures).
-   **LLM-Powered `Fine Split`**: Instead of relying on naive heuristics, SequenceFormer uses an LLM to perform a `Fine Split`, intelligently dividing the document into semantically complete `Atom Chunks`.
-   **Multi-Modal Enrichment**: SequenceFormer can leverage Vision-Language Models (VLMs) in a `Post-Enrichment` step to analyze embedded images and integrate their descriptions, creating a true multi-modal representation.
-   **Streaming & Stateful**: The engine processes documents as a stream, maintaining context (like hierarchical headings) between segments to ensure global consistency.

## Features

-   **Intelligent Text Chunking**: Moves beyond fixed-size or rule-based splitting to create semantically meaningful chunks.
-   **Hierarchical Heading Detection**: Automatically detects and maintains the document's heading structure (e.g., `# Title`, `## Section 1`).
-   **Atomic Content Protection**: Guarantees that fenced content blocks (like code, diagrams, or tables) are never split, preserving their integrity.
-   **Multi-Modal VLM Integration**: Can analyze embedded images, generate descriptions, and reconstruct the content to include both text and image analysis in a structured format.
-   **Customizable Metadata Extraction**: Use a simple JSON schema to instruct the LLM to classify and tag each chunk according to your specific needs (e.g., by document type, form, or any other category).
-   **Layout-Aware Processing with `minerU`**: Natively supports the output format of `minerU`, allowing it to process complex documents (like PDFs) with an understanding of their original layout, including text blocks and images.
-   **Small Chunk Merging**: Automatically merges chunks that are too small with adjacent chunks under the same heading, ensuring that all final chunks meet a minimum size threshold for better semantic coherence.
-   **Streaming Output**: Produces a JSONL stream of `Atom Chunks`, ideal for immediate ingestion into downstream systems like vector databases.
-   **Robust & Configurable**: Highly configurable through CLI arguments, `.env` files, and a global config.

## Installation

```bash
# Coming soon to PyPI!
pip install sequence-former 
```

## Usage

### Command-Line Interface (CLI)

The primary way to use SequenceFormer is through its command-line interface.

```bash
sequence-former [INPUT_PATH] [OPTIONS]
```

**Example: Basic Text Processing**

This command processes a Markdown file, automatically using the default structural metadata schema, and prints the resulting JSONL to the console.

```bash
sequence-former "examples/sample_documents/design_document.md"
```

**Example: Multi-Modal Processing**

This command processes a directory output by `minerU` (containing a `.md` file and an `images/` folder), enables VLM enrichment for the images, and saves the output to a file.

```bash
sequence-former "examples/sample_documents/KIMI_K2_Truncated_MinerU/" --enable-vlm -o "examples/sample_outputs/KIMI_K2_enriched.jsonl"
```

**Example: Custom Metadata Extraction**

This command uses a custom schema to classify chunks based on project-specific criteria.

```bash
sequence-former "path/to/my_document.md" --metadata-schema "path/to/my_schema.json"
```

**Example: Merging Small Chunks**

This command processes a document and ensures that any chunk smaller than 256 characters is merged with an adjacent chunk under the same heading.

```bash
sequence-former "path/to/my_document.md" --min_chunk_size 256
```

### Python Library

You can also integrate SequenceFormer directly into your Python applications.

```python
from sequence_former.client import SequenceFormerClient
from sequence_former.config import load_settings

# 1. Load settings from files, environment, or defaults
# CLI arguments can be passed as a dictionary
cli_args = {
    "input_path": "path/to/your/doc.md",
    "output": "path/to/output.jsonl",
    "enable_vlm": True
}
settings = load_settings(cli_args)

# 2. Instantiate the client
client = SequenceFormerClient(settings)

# 3. Process the file
client.process_file(settings.input_path, settings.output)

print("Processing complete!")

# Note: The process_text(text: str) -> List[Chunk] method for in-memory processing
# is planned for a future release.
```

## Configuration

SequenceFormer uses a hierarchical configuration system with the following priority:

1.  **Command-Line Arguments**: Highest priority (e.g., `--openai_model gpt-4o`).
2.  **.env File**: In your project's root directory.
3.  **Global Config File**: Located at `~/.sequence_former/settings.json`.
4.  **Environment Variables**: Lowest priority.

### Metadata Schema

The `--metadata-schema` option allows you to provide a JSON file that defines the classification categories for the LLM. The keys are the categories, and the values are lists of allowed options.

**Example: `default_metadata_schema.json`**
```json
{
  "type": [
    "Title", "Abstract", "Introduction", "Methodology", "Results", "Conclusion", ...
  ],
  "form": [
    "text", "list", "figure", "table", "formula", "code"
  ]
}
```

This is a powerful feature to tailor the output to your specific RAG or data analysis needs.
