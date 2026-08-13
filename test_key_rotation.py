#!/usr/bin/env python3
"""Test to verify the key rotation mechanism works correctly."""

import sys
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, "/workspaces/BITS_programming/model-engineering/src")

from models.gemini_client import GeminiClient


def test_key_rotation_on_429():
    """Verify that the client rotates keys when it encounters a 429 error."""
    
    # Mock genai to avoid needing real API keys
    with patch('models.gemini_client.genai') as mock_genai:
        # Setup: Create a mock client that will fail on first two attempts with 429
        mock_client_instance = MagicMock()
        mock_genai.Client.return_value = mock_client_instance
        
        # First call: simulate 429 error
        # Second call: simulate 429 error again
        # Third call: simulate success
        mock_client_instance.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED: Quota exceeded"),
            Exception("429 RESOURCE_EXHAUSTED: Quota exceeded"),
            MagicMock(
                text="Success on third attempt",
                usage_metadata=None,
                candidates=[]
            )
        ]
        
        # Create client with multiple keys
        with patch.dict('os.environ', {
            'GEMINI_API_KEY': 'key1',
            'GEMINI_API_KEY_1': 'key2',
            'GEMINI_API_KEY_2': 'key3',
        }):
            client = GeminiClient()
            
            print("✓ Client initialized with 3 keys")
            print(f"  Keys: {client.api_keys}")
            print(f"  Current key index: {client._key_index}")
            print(f"  Current key: {client.api_key}")
            
            # Make a call that will trigger retries with rotation
            response = client.generate("Test prompt")
            
            print("\n✓ Generate call completed")
            print(f"  Response: {response.text}")
            print(f"  Final key index: {client._key_index}")
            print(f"  Final key: {client.api_key}")
            
            # Verify that genai.Client was called multiple times with different keys
            print("\n✓ Verification:")
            print(f"  genai.Client was called {mock_genai.Client.call_count} times")
            
            calls = mock_genai.Client.call_args_list
            print(f"  Call sequence:")
            for i, call in enumerate(calls):
                kwargs = call[1]
                key = kwargs.get('api_key', 'unknown')
                print(f"    Attempt {i+1}: Created client with key={key[:4]}...")
            
            # Verify key rotation happened
            assert mock_genai.Client.call_count == 3, "Should create 3 clients (initial + 2 rotations)"
            
            # Check the API keys used
            api_keys_used = [call[1]['api_key'] for call in calls]
            assert api_keys_used[0] == 'key1', "Should start with key1"
            assert api_keys_used[1] == 'key2', "Should rotate to key2 on first 429"
            assert api_keys_used[2] == 'key3', "Should rotate to key3 on second 429"
            
            print("\n✅ KEY ROTATION IS WORKING CORRECTLY!")
            print("   - Started with key1")
            print("   - Rotated to key2 after first 429")
            print("   - Rotated to key3 after second 429")
            print("   - Succeeded on third attempt")


if __name__ == "__main__":
    test_key_rotation_on_429()
