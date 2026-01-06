import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

# Mocking environment variables
with patch.dict('os.environ', {
    'META_ACCESS_TOKEN': 'fake_token',
    'META_AD_ACCOUNT_ID': '123',
    'META_CAMPAIGN_ID': '456'
}):
    import importlib
    performance_score = importlib.import_module("Performance Score")

class TestPerformanceBatch(unittest.TestCase):
    @patch('requests.post')
    def test_fetch_ad_metadata_batch(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "code": 200,
                "body": '{"id": "ad1", "effective_status": "ACTIVE", "created_time": "2023-01-01T00:00:00+0000"}'
            }
        ]
        mock_post.return_value = mock_response

        ad_ids = ["ad1"]
        df = performance_score.fetch_ad_metadata(ad_ids)
        
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["ad_id"], "ad1")
        self.assertEqual(df.iloc[0]["effective_status"], "ACTIVE")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
