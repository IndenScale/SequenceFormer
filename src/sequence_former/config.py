import os
import json
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

class Settings(BaseModel):
    """
    Manages all application settings with a hierarchical loading mechanism.
    """
    # --- LLM Provider ---
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key for text processing")
    openai_base_url: Optional[str] = Field(None, description="OpenAI compatible base URL for text processing")
    openai_model: str = Field("gpt-4-turbo", description="The text model to use for processing")

    # --- VLM Provider (new) ---
    openai_vlm_api_key: Optional[str] = Field(None, description="OpenAI API Key for vision processing")
    openai_vlm_base_url: Optional[str] = Field(None, description="OpenAI compatible base URL for vision processing")
    openai_vlm_model: str = Field("gpt-4-vision-preview", description="The vision model to use for image analysis")

    # --- Processing Parameters ---
    long_doc_chunk_size: int = Field(8000, description="The character size for pre-splitting long documents")
    target_chunk_size: int = Field(1024, description="The ideal target character size for LLM-generated chunks.")
    chunk_size_tolerance: float = Field(0.1, description="Tolerance for chunk size deviation (e.g., 0.1 for 10% tolerance).")
    min_chunk_size: int = Field(256, description="The minimum character size for a chunk. Chunks smaller than this will be merged.")
    
    # --- Metadata Management ---
    metadata_schema_path: Optional[str] = Field(None, description="Path to a JSON file containing the metadata extraction schema.")
    
    # --- CLI & Logging ---
    input_path: Optional[str] = Field(None, description="Input document/directory path (from CLI)")
    output: str = Field("-", description="Output file path (from CLI, '-' for stdout)")
    debug: bool = Field(False, description="Enable debug logging (from CLI)")
    retain_intermediate: bool = Field(False, description="Retain intermediate files for debugging.")
    enable_vlm: bool = Field(False, description="Enable VLM processing for embedded images.")
    mineru: bool = Field(False, description="Enable MinerU layout-aware processing.")

def load_settings(args: Optional[dict] = None) -> Settings:
    """
    Loads settings from multiple sources with a defined priority.
    Priority: CLI args > .env file > Global config > Environment variables
    """
    # 1. Load from environment variables (lowest priority)
    env_settings = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "openai_vlm_api_key": os.getenv("OPENAI_VLM_API_KEY"),
        "openai_vlm_base_url": os.getenv("OPENAI_VLM_BASE_URL"),
        "openai_vlm_model": os.getenv("OPENAI_VLM_MODEL"),
    }
    env_settings = {k: v for k, v in env_settings.items() if v is not None}
    
    # 2. Load from global config file
    global_config_path = os.path.expanduser("~/.sequence_former/settings.json")
    global_settings = {}
    if os.path.exists(global_config_path):
        with open(global_config_path, 'r') as f:
            global_settings = json.load(f)

    # 3. Load from .env file (overwrites global and env vars)
    load_dotenv() 
    dotenv_settings = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "openai_vlm_api_key": os.getenv("OPENAI_VLM_API_KEY"),
        "openai_vlm_base_url": os.getenv("OPENAI_VLM_BASE_URL"),
        "openai_vlm_model": os.getenv("OPENAI_VLM_MODEL"),
    }
    dotenv_settings = {k: v for k, v in dotenv_settings.items() if v is not None}

    combined_settings = {**env_settings, **global_settings, **dotenv_settings}

    # 4. Load from CLI arguments (highest priority)
    cli_settings = args if args else {}
    cli_settings = {k: v for k, v in cli_settings.items() if v is not None}
    
    final_settings_data = {**combined_settings, **cli_settings}
    
    return Settings(**final_settings_data)

def ensure_global_config_dir_exists():
    """Ensures the global config directory exists."""
    dir_path = os.path.expanduser("~/.sequence_former")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created global settings directory at: {dir_path}")