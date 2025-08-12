import json
import logging
import base64
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError

from .data_models import ProcessingState, LLMOutput
from .config import Settings

def _format_page_for_prompt(enriched_page: List[Dict[str, Any]]) -> str:
    """Formats the enriched page into a string that the LLM can reference."""
    lines = []
    for line in enriched_page:
        lines.append(f"p{line['page']}:l{line['line']}:{line['text']}")
    return "\n".join(lines)

def build_prompt(enriched_page: List[Dict[str, Any]], state: ProcessingState, settings: Settings) -> str:
    """
    Builds the full prompt for the text LLM call.
    """
    # ... (This function remains the same as before)
    instruction = f"""You are an expert document analyst. Your task is to intelligently split the provided text into semantically coherent chunks and extract structured information based on a template.

**CRITICAL RULE: Fenced content blocks (e.g., ```python, ```mermaid, ```json) are atomic and MUST NOT be split. If a fenced block is large, it should be placed in its own chunk, even if this causes the chunk to exceed the ideal character length.**

Each chunk you create should ideally be around {settings.target_chunk_size} characters long. The length of the raw text for each chunk must not exceed {int(settings.target_chunk_size * (1 + settings.chunk_size_tolerance))} characters, unless a single fenced block requires it.
"""
    output_format_instruction = """You MUST respond with a single, valid JSON object. Do NOT add any text, explanations, or markdown formatting before or after the JSON object.
Your entire response must be only the JSON.
The JSON object must have two keys: "hierarchical_headings" and "chunks".
- "hierarchical_headings" must be a list of strings representing the document's titles and subtitles.
- "chunks" must be a list of objects, where each object has keys like "start_page", "start_line", "end_page", "end_line", "summary", and "metadata".
"""
    metadata_schema_instruction = ""
    if state.metadata_schema:
        metadata_schema_instruction = f"""
--- STRUCTURED METADATA EXTRACTION ---
For each chunk, you must fill out the "metadata" field.
The metadata you extract must conform to the following JSON schema of categories and allowed values.

**Metadata Schema:**
```json
{json.dumps(state.metadata_schema, indent=2, ensure_ascii=False)}
```

**Rules for Extraction:**
- For each category in the schema, choose the most appropriate value(s) from the provided list.
- If a category is not applicable to a chunk, omit it from that chunk's "metadata" dictionary.
"""
    context_instruction = f"""
--- DOCUMENT HIERARCHY CONTEXT ---
The document's hierarchical heading structure identified so far is:
{state.hierarchical_headings}

Update this list with any new headings found in the current text content.
- Use Markdown heading styles (e.g., '#', '##', '###').
- The first heading in the list is derived from the filename. If the content suggests a more appropriate title for the document, you may correct this first heading.
"""
    formatted_content = _format_page_for_prompt(enriched_page)
    content_section = f"""
--- DOCUMENT CONTENT TO PROCESS ---
{formatted_content}
--- END OF CONTENT ---
"""
    final_prompt = f"""{instruction}
{output_format_instruction}
{metadata_schema_instruction}
{context_instruction}
{content_section}
Your JSON response:
"""
    return final_prompt

def call_llm(prompt: str, settings: Settings) -> str:
    """
    Calls an OpenAI-compatible API for text processing.
    """
    # ... (This function remains the same)
    if not settings.openai_api_key:
        logging.error("OPENAI_API_KEY is not configured.")
        raise ValueError("API key not found.")
    try:
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        logging.debug(f"Calling model {settings.openai_model}...")
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logging.debug("Successfully received response from text LLM.")
        return content if content else "{}"
    except APIError as e:
        logging.error(f"An API error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during the text LLM call: {e}")
        raise

def _encode_image_to_base64(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to read or encode image at {image_path}: {e}")
        raise

def call_vlm(image_path: str, text_context: str, settings: Settings) -> str:
    """
    Calls an OpenAI-compatible Vision-Language Model (VLM) with an image and its context.
    """
    if not settings.openai_vlm_api_key:
        logging.error("OPENAI_VLM_API_KEY is not configured.")
        raise ValueError("VLM API key not found.")

    try:
        base64_image = _encode_image_to_base64(image_path)
        
        client = OpenAI(api_key=settings.openai_vlm_api_key, base_url=settings.openai_vlm_base_url)
        
        vlm_prompt = f"""The user has provided an image and the surrounding text from a document.
Your task is to provide a detailed, comprehensive description of the image's content, paying close attention to any text, data, or diagrams shown.
Explain what the image represents in the context of the surrounding text.
The surrounding text is:
---
{text_context}
---
Your description:"""

        logging.debug(f"Calling VLM model {settings.openai_vlm_model} for image: {image_path}")
        response = client.chat.completions.create(
            model=settings.openai_vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        description = response.choices[0].message.content
        logging.debug("Successfully received description from VLM.")
        return description if description else "No description generated."

    except APIError as e:
        logging.error(f"A VLM API error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during the VLM call: {e}")
        raise

def parse_llm_output(json_string: str) -> LLMOutput:
    """
    Parses the JSON string from the text LLM.
    """
    # ... (This function remains the same)
    try:
        return LLMOutput.model_validate_json(json_string)
    except Exception as e:
        logging.error(f"Failed to parse or validate LLM JSON output: {e}")
        logging.debug(f"Invalid JSON string: {json_string}")
        raise

def process_page_with_llm(enriched_page: List[Dict[str, Any]], state: ProcessingState, settings: Settings) -> LLMOutput:
    """
    Orchestrates the text processing of a single page.
    """
    # ... (This function remains the same)
    prompt = build_prompt(enriched_page, state, settings)
    logging.debug(f"Generated Prompt:\n{prompt}") 
    llm_json_response = call_llm(prompt, settings)
    llm_output = parse_llm_output(llm_json_response)
    return llm_output