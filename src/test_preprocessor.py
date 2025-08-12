# src/test_preprocessor.py
import unittest
import os
import shutil
from .preprocessor import load_and_enrich_document

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.base_dir = os.path.dirname(__file__)
        self.test_data_dir = os.path.abspath(os.path.join(self.base_dir, '..', 'data', 'input'))
        self.paginated_dir = os.path.abspath(os.path.join(self.base_dir, '..', 'data', 'test_paginated_for_enrich'))
        
        os.makedirs(self.paginated_dir, exist_ok=True)
        
        for i in range(1, 4):
            shutil.copy(
                os.path.join(self.test_data_dir, f'multi_page_p{i}.txt'),
                os.path.join(self.paginated_dir, f'page_{i}.txt')
            )

    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.paginated_dir)

    def test_enrich_paginated_docs(self):
        """Tests that a directory of files is correctly enriched."""
        doc = load_and_enrich_document(self.paginated_dir)
        
        self.assertIsInstance(doc, list)
        self.assertEqual(len(doc), 3) # 3 pages
        
        page1 = doc[0]
        self.assertIsInstance(page1, list)
        self.assertIsInstance(page1[0], dict)
        
        # Check first line of page 1
        self.assertEqual(page1[0]['page'], 1)
        self.assertEqual(page1[0]['line'], 1)
        self.assertEqual(page1[0]['text'], '# The Grand Design')
        
        # Check first line of page 3
        page3 = doc[2]
        self.assertEqual(page3[0]['page'], 3)
        self.assertEqual(page3[0]['line'], 1)
        self.assertIn('Indeed, the implications', page3[0]['text'])

    def test_enrich_long_doc(self):
        """Tests that a long file is correctly split and enriched."""
        long_doc_path = os.path.join(self.test_data_dir, 'long_document.txt')
        
        doc = load_and_enrich_document(long_doc_path, long_doc_chunk_size=1000)
        
        self.assertIsInstance(doc, list)
        self.assertTrue(len(doc) > 1) # Should be split into multiple pages/chunks
        
        page1 = doc[0]
        self.assertIsInstance(page1, list)
        self.assertIsInstance(page1[0], dict)
        
        # Check first line of first chunk (page 1)
        self.assertEqual(page1[0]['page'], 1)
        self.assertEqual(page1[0]['line'], 1)
        self.assertEqual(page1[0]['text'], '# A Long Treatise on Digital Gardens')
        
        # Check first line of second chunk (page 2)
        page2 = doc[1]
        self.assertEqual(page2[0]['page'], 2)
        self.assertEqual(page2[0]['line'], 1)

if __name__ == '__main__':
    unittest.main()