import json
import unittest
from unittest.mock import MagicMock, patch

# Mocking environment variables before importing the script
with patch.dict('os.environ', {
    'META_ACCESS_TOKEN': 'fake_token',
    'META_AD_ACCOUNT_ID': '123',
    'META_CAMPAIGN_ID': '456'
}):
    import download_creatives

class TestBatchLogic(unittest.TestCase):
    @patch('download_creatives.session.post')
    def test_graph_batch_get(self, mock_post):
        # Configura o mock para retornar uma resposta de sucesso
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"code": 200, "body": '{"id": "123", "name": "test"}'}]
        mock_post.return_value = mock_response

        batch = [{"method": "GET", "relative_url": "123"}]
        result = download_creatives.graph_batch_get(batch, "fake_token")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["code"], 200)
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
