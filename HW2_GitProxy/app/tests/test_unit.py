import pytest
import httpx
import json
import hmac
import hashlib
import os
import sys
from typing import Callable
from unittest import mock 

#We patch os.environ with temporary values that satisfy pydantic's requirements
# This must happen BEFORE the import chain starts loading modules that use pydantic (like config.py)
MOCK_ENV_VARS = {
    "GITHUB_TOKEN": "mock-token-123",
    "GITHUB_OWNER": "test_owner",
    "GITHUB_REPO": "test_repo",
    "WEBHOOK_SECRET": "mock-secret-456",
}

# Start env patch BEFORE importing modules that read it
_env_patcher = mock.patch.dict(os.environ, MOCK_ENV_VARS, clear=False)
_env_patcher.start()

import atexit
atexit.register(_env_patcher.stop)

# Create a mock for the 'utils' module and the function it provides
def mock_parse_pagination_headers(response):
    # Always return a default dictionary for the test
    return {} 
mock_utils = mock.Mock()
mock_utils.parse_pagination_headers = mock_parse_pagination_headers

# Inject the mock 'utils' module into Python's module registry before importing GitHubClient
sys.modules['utils'] = mock_utils

# Import the code we are testing
# NOTE: Ensure these exceptions are defined and imported correctly
from github_client import GitHubClient, NotFoundError, AuthError, BadRequestError 
from webhook_security import verify_signature, generate_valid_signature


# --- Setup: Helper to build mocked responses for httpx ---

def mock_response_builder(status_code: int = 200, content: dict = None, url: str = "http://mocked-url", headers: dict = None):
    """Creates a basic httpx.Response object."""
    return httpx.Response(
        status_code=status_code,
        json=content if content is not None else {},
        request=httpx.Request("GET", url), 
        headers=headers
    )

class AsyncMockTransport(httpx.AsyncBaseTransport):
    """An asynchronous transport that simply returns a predefined response."""
    def __init__(self, response: httpx.Response):
        self.response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """This method fulfills the httpx.AsyncBaseTransport interface."""
        # This handles the call from httpx.AsyncClient and returns the mocked response
        return self.response

# --- Internal Helper for URL Cleaning ---
def get_clean_base_url(client: GitHubClient) -> str:
    """
    Extracts the base_url from the client and cleans up common duplication issues.
    The client uses `self._client` for its httpx.AsyncClient instance.
    """
    # Use client._client instead of client.client
    raw_url = str(client._client.base_url) 
    # Clean up the URL to ensure it doesn't contain duplicated components
    return raw_url.replace("/repos/repos/", "/repos/").rstrip('/')


# --- UNIT TESTS: GitHub Client Error Mapping ---

@pytest.mark.asyncio
async def test_error_mapping_404_not_found():
    """
    Tests if a GitHub 404 response is correctly translated into a NotFoundError.
    """
    client = GitHubClient()
    
    # 2. Derive the expected full URL using the client's internal base_url AND cleaning it
    clean_base_url = get_clean_base_url(client)
    # The client constructs the full URL path: base_url + /repos/{owner}/{repo}/issues/{number}
    expected_full_url = f"{clean_base_url}/repos/{client.owner}/{client.repo}/issues/999"

    mock_response = mock_response_builder(
        status_code=404, 
        url=expected_full_url,
        content={"message": "Resource not found."} 
    )

    mock_transport = AsyncMockTransport(mock_response)
    
    # Inject the mocked transport into the client's internal httpx client (_client)
    client._client._transport = mock_transport 
    
    # 6. Assert the Exception: MUST AWAIT THE CALL
    with pytest.raises(NotFoundError) as excinfo:
        await client.get_issue(999) 
    
    assert "not found" in str(excinfo.value).lower()

@pytest.mark.asyncio
async def test_error_mapping_401_auth_error():
    """
    Tests if a GitHub 401 response (invalid token) is translated into an AuthError.
    """
    client = GitHubClient()
    
    clean_base_url = get_clean_base_url(client)
    expected_full_url = f"{clean_base_url}/repos/{client.owner}/{client.repo}/issues/1"
    
    mock_response = mock_response_builder(
        status_code=401, 
        url=expected_full_url,
        content={"message": "Bad credentials"} 
    )

    mock_transport = AsyncMockTransport(mock_response)
    
    client._client._transport = mock_transport 
    
    # 6. Assert the Exception: MUST AWAIT THE CALL
    with pytest.raises(AuthError) as excinfo:
        await client.get_issue(1) 
    
    assert "authentication failed" in str(excinfo.value).lower()

@pytest.mark.asyncio
async def test_error_mapping_422_bad_request():
    """
    Tests if a GitHub 422 response (validation error) is translated into a BadRequestError.
    """
    client = GitHubClient()
    
    clean_base_url = get_clean_base_url(client)
    expected_full_url = f"{clean_base_url}/repos/{client.owner}/{client.repo}/issues/1"
    
    mock_response = mock_response_builder(
        status_code=422, 
        url=expected_full_url,
        content={"message": "Validation Failed", "errors": []}
    )

    mock_transport = AsyncMockTransport(mock_response)
    
    client._client._transport = mock_transport  
        
    # 6. Assert the Exception: MUST AWAIT THE CALL
    with pytest.raises(BadRequestError) as excinfo:
        await client.get_issue(1)
    
    assert "validation failed" in str(excinfo.value).lower()


# --- UNIT TESTS: Webhook Security Verification ---

TEST_PAYLOAD = json.dumps({"action": "opened", "issue": {"number": 1, "title": "Test Issue"}}).encode('utf-8')

def test_webhook_valid_signature():
    """
    Asserts that the function returns True for a correctly calculated signature.
    """
    valid_signature = generate_valid_signature(TEST_PAYLOAD)
    is_valid = verify_signature(TEST_PAYLOAD, valid_signature)
    assert is_valid is True

def test_webhook_invalid_secret_signature():
    """
    Asserts that the function returns False if the payload was signed with a different secret (simulates a bad sender).
    """
    fake_secret = "a_different_secret_987"
    wrong_hmac = hmac.new(fake_secret.encode('utf-8'), TEST_PAYLOAD, hashlib.sha256).hexdigest()
    wrong_signature = f"sha256={wrong_hmac}"
    
    is_valid = verify_signature(TEST_PAYLOAD, wrong_signature)
    
    assert is_valid is False

def test_webhook_tampered_body():
    """
    Asserts that the function returns False if the body was changed after signing 
    (simulates an attacker modifying the data in transit).
    """
    valid_signature = generate_valid_signature(TEST_PAYLOAD)
    
    tampered_payload = json.dumps({"action": "opened", "issue": {"number": 1, "title": "HACKED TITLE"}}).encode('utf-8')
    
    is_valid = verify_signature(tampered_payload, valid_signature)
    
    assert is_valid is False

def test_webhook_missing_signature_header():
    """
    Tests handling of requests with a completely missing signature header.
    """
    is_valid = verify_signature(TEST_PAYLOAD, signature_header="")
    assert is_valid is False
    
def test_webhook_invalid_signature_format():
    """
    Tests handling of requests with a malformed signature header.
    """
    is_valid = verify_signature(TEST_PAYLOAD, signature_header="invalid_format_no_equals")
    assert is_valid is False
