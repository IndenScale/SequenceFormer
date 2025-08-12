from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict, Any

class Chunk(BaseModel):
    """
    Defines a semantically coherent text block (Chunk).
    """
    start_page: int = Field(..., description="Starting page number")
    start_line: int = Field(..., description="Starting line number")
    end_page: int = Field(..., description="Ending page number")
    end_line: int = Field(..., description="Ending line number")
    
    summary: str = Field(..., description="A summary of the content of this chunk")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured metadata extracted based on the provided schema.")
    
    heading: Optional[str] = Field(None, description="The most recent heading this chunk belongs to")
    raw_text: Optional[str] = Field(None, description="The original raw text content, populated during post-processing")

class LLMOutput(BaseModel):
    """
    Defines the expected structure of the LLM's output for a single processing call.
    """
    hierarchical_headings: List[str] = Field(..., description="An updated list of hierarchical headings, from highest to lowest level")
    chunks: List[Chunk] = Field(..., description="A list of chunks extracted from the current content")

class ProcessingState(BaseModel):
    """
    Defines the state that is passed between the processing of different pages/sections.
    """
    doc_id: str = Field(..., description="A unique identifier for the document being processed")
    hierarchical_headings: List[str] = Field(default_factory=list, description="The currently known hierarchical headings")
    
    # Renamed to metadata_schema to reflect its purpose as a static template.
    metadata_schema: Dict[str, Any] = Field(default_factory=dict, description="The schema/template for metadata extraction.")
    
    staged_chunk: Optional[Chunk] = Field(None, description="A chunk from the previous page that was incomplete and is staged for reconsideration")
    unprocessed_lines: List[Dict[str, Any]] = Field(default_factory=list, description="Lines from the previous batch that were not assigned to any chunk.")
    processed_lines: Set[str] = Field(default_factory=set, description="A set of line identifiers (e.g., 'p{page}_l{line}') that have been archived into a chunk")