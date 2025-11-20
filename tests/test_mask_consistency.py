# tests/test_mask_consistency.py
import unittest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMaskConsistency(unittest.TestCase):

    def test_mask_and_index_consistency(self):
        """
        Tests that for every sample in a cleaned dataset, the policy_index
        corresponds to a True value in the legal_moves_mask.
        """
        cleaned_file_path = "tests/dirty_data_cleaned.jsonl"
        self.assertTrue(os.path.exists(cleaned_file_path), f"Cleaned data file not found at {cleaned_file_path}")

        with open(cleaned_file_path, 'r') as f:
            for i, line in enumerate(f):
                record = json.loads(line)

                policy_index = record.get('policy_index')
                legal_moves_mask = record.get('legal_moves_mask')

                self.assertIsNotNone(policy_index, f"Line {i+1}: policy_index is missing.")
                self.assertIsNotNone(legal_moves_mask, f"Line {i+1}: legal_moves_mask is missing.")

                self.assertIsInstance(policy_index, int)
                self.assertIsInstance(legal_moves_mask, list)
                self.assertEqual(len(legal_moves_mask), 4096)

                self.assertTrue(
                    legal_moves_mask[policy_index],
                    f"Line {i+1}: Inconsistency found! policy_index ({policy_index}) is not a legal move according to the mask."
                )

if __name__ == '__main__':
    unittest.main()
