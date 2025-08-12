# src/test_llm_processor.py
import unittest
from .data_models import ProcessingState, LLMOutput
from .llm_processor import build_prompt, process_page_with_llm

class TestLLMProcessor(unittest.TestCase):

    def setUp(self):
        """Set up a mock enriched page and processing state."""
        self.mock_page = [
            {'page': 1, 'line': 1, 'text': '# Chapter 1'},
            {'page': 1, 'line': 2, 'text': ''},
            {'page': 1, 'line': 3, 'text': 'This is the first paragraph.'},
            {'page': 1, 'line': 4, 'text': 'It has two sentences.'},
        ]
        self.mock_state = ProcessingState(
            doc_id="test_doc",
            hierarchical_headings=["# Document Title"]
        )

    def test_build_prompt(self):
        """Tests that the generated prompt contains all necessary components."""
        prompt = build_prompt(self.mock_page, self.mock_state)

        # 1. Check for core instruction
        self.assertIn("You are an expert document analyst.", prompt)
        
        # 2. Check for output format instruction
        self.assertIn("You MUST respond with a single, valid JSON object", prompt)
        self.assertIn("LLMOutput", prompt) # Check that the schema name is mentioned

        # 3. Check for context (headings)
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("# Document Title", prompt)

        # 4. Check for formatted content
        self.assertIn("--- DOCUMENT CONTENT TO PROCESS ---", prompt)
        self.assertIn("p1:l1:# Chapter 1", prompt)
        self.assertIn("p1:l4:It has two sentences.", prompt)

    def test_build_prompt_with_tags(self):
        """Tests that the tag dictionary is correctly included in the prompt."""
        tags = ["Technology", "Science", "History"]
        prompt = build_prompt(self.mock_page, self.mock_state, tag_dictionary=tags)
        
        self.assertIn("Tag Dictionary:", prompt)
        self.assertIn("Technology", prompt)
        self.assertIn("History", prompt)

    def test_process_page_with_llm(self):
        """
        Tests the full orchestration function.
        It uses a mock LLM call, so this mainly tests the integration and parsing.
        """
        llm_output = process_page_with_llm(self.mock_page, self.mock_state)

        self.assertIsInstance(llm_output, LLMOutput)
        
        # Check headings from the mock response
        self.assertIn("# A Long Treatise on Digital Gardens", llm_output.hierarchical_headings)
        
        # Check chunks from the mock response
        self.assertEqual(len(llm_output.chunks), 2)
        self.assertEqual(llm_output.chunks[0].summary, "Introduction to the concept of a digital garden as a metaphor for cultivating interconnected knowledge, contrasting it with traditional blogs.")
        self.assertEqual(llm_output.chunks[0].start_page, 1)
        self.assertEqual(llm_output.chunks[0].start_line, 1)

if __name__ == '__main__':
    unittest.main()
