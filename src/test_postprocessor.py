# src/test_postprocessor.py
import unittest
from .data_models import ProcessingState, LLMOutput, Chunk
from .postprocessor import finalize_chunks_and_update_state

class TestPostProcessor(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.initial_state = ProcessingState(
            doc_id="test_doc",
            hierarchical_headings=["# Old Heading"],
            staged_chunk=None,
            processed_lines=set()
        )

        # This represents the full text fed to the LLM
        self.combined_lines = [
            {'page': 1, 'line': 1, 'text': '# Chapter 1'},
            {'page': 1, 'line': 2, 'text': 'First paragraph.'},
            {'page': 1, 'line': 3, 'text': '## Section 1.1'},
            {'page': 1, 'line': 4, 'text': 'Second paragraph, continues...'},
            # Page boundary would be here in a real scenario
            {'page': 2, 'line': 1, 'text': '...continuation of second paragraph.'},
            {'page': 2, 'line': 2, 'text': 'Third paragraph, potentially incomplete.'}
        ]
        
        # This is the mock output from the LLM for the `combined_lines`
        self.llm_output = LLMOutput(
            hierarchical_headings=["# Chapter 1", "## Section 1.1"],
            chunks=[
                Chunk(start_page=1, start_line=1, end_page=1, end_line=2, summary="Intro", tags=[]),
                Chunk(start_page=1, start_line=3, end_page=2, end_line=1, summary="Middle", tags=[]),
                Chunk(start_page=2, start_line=2, end_page=2, end_line=2, summary="End", tags=[])
            ]
        )

    def test_finalize_chunks_and_update_state(self):
        """
        Tests the core logic of finalizing chunks and updating the processing state.
        """
        # We are processing page 2, so the boundary is 2.
        current_page_boundary = 2
        
        reliable_chunks, new_state = finalize_chunks_and_update_state(
            self.llm_output,
            self.combined_lines,
            current_page_boundary,
            self.initial_state
        )

        # 1. Verify reliable chunks
        # The first two chunks should be reliable, the last one should be staged.
        self.assertEqual(len(reliable_chunks), 2)
        self.assertEqual(reliable_chunks[0].summary, "Intro")
        self.assertEqual(reliable_chunks[1].summary, "Middle")
        
        # Check that raw_text was populated for reliable chunks
        self.assertIn("# Chapter 1", reliable_chunks[0].raw_text)
        self.assertIn("First paragraph.", reliable_chunks[0].raw_text)
        self.assertIn("...continuation of second paragraph.", reliable_chunks[1].raw_text)

        # 2. Verify the new state
        self.assertIsInstance(new_state, ProcessingState)
        
        # Check for updated headings
        self.assertEqual(new_state.hierarchical_headings, ["# Chapter 1", "## Section 1.1"])
        
        # Check for the new staged chunk
        self.assertIsNotNone(new_state.staged_chunk)
        self.assertEqual(new_state.staged_chunk.summary, "End")
        
        # Check that processed lines have been updated correctly
        expected_processed_lines = {
            # From chunk 1
            "p1:l1", "p1:l2",
            # From chunk 2 (which is now handled correctly)
            "p1:l3", "p1:l4",
            "p2:l1"
        }
        self.assertEqual(new_state.processed_lines, expected_processed_lines)


if __name__ == '__main__':
    unittest.main()
