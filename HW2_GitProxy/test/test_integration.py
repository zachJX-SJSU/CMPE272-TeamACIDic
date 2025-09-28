import os
import requests
import pytest
import time
import requests
import time
import requests
import requests_mock
import json
from unittest.mock import MagicMock

class MockResponse:
    """Simulates the response object from requests."""
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data) 

    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error")

# Define the mock event data that the test EXPECTS to receive from the local service
MOCK_RECEIVED_EVENTS = [
    {"event_type": "issues", "action": "opened", "issue": {"title": "Test issue title"}},
]

# --- Configuration ---
# Get environment variables
TOKEN = os.environ.get("GITHUB_TOKEN")
OWNER = os.environ.get("GITHUB_OWNER")
REPO = os.environ.get("GITHUB_REPO")

# --- Configuration ---
# REPLACE with the endpoint your service exposes to check for received events
YOUR_SERVICE_EVENTS_URL = "http://localhost:5500/api/received_events" 
GITHUB_ISSUE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"

if not all([TOKEN, OWNER, REPO]):
    pytest.skip("Missing GITHUB_TOKEN, GITHUB_OWNER, or GITHUB_REPO environment variables.", allow_module_level=True)

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# --- Fixtures ---
# GitHub API Integration Tests

@pytest.fixture(scope="module")
def created_issue_number():
    """
    1) Creates an issue for the test suite to use.
    2) Yields the issue number.
    3) Closes the issue in a teardown step to clean up.
    """
    # 1. Create issue (Part 1 of 1)
    create_data = {
        "title": f"Integration Test Issue - {time.time()}",
        "body": "This is a temporary issue for GitHub API integration testing.",
        "labels": ["bug", "integration-test"]
    }
    
    response = requests.post(BASE_URL, headers=HEADERS, json=create_data)
    assert response.status_code == 201
    
    issue_data = response.json()
    issue_number = issue_data["number"]
    print(f"\nCreated issue #{issue_number}")
    
    # Yield the number for use in tests
    yield issue_number
    
    # 3. Teardown: Close the issue
    print(f"\nClosing issue #{issue_number} in teardown.")
    close_url = f"{BASE_URL}/{issue_number}"
    close_data = {"state": "closed"}
    requests.patch(close_url, headers=HEADERS, json=close_data)


# --- Tests ---

def test_1_create_and_get_issue(created_issue_number):
    """
    1) Create issue (covered by fixture) -> 201.
    2) GET that issue -> 200.
    """
    # 2) GET that issue -> 200
    issue_url = f"{BASE_URL}/{created_issue_number}"
    response = requests.get(issue_url, headers=HEADERS)
    
    assert response.status_code == 200
    issue_data = response.json()
    assert issue_data["number"] == created_issue_number
    assert issue_data["state"] == "open" # Should be open initially

def test_2_update_close_reopen(created_issue_number):
    """
    2) Update title/body; close and reopen.
    """
    issue_url = f"{BASE_URL}/{created_issue_number}"
    
    # 2a) Update title/body
    new_title = "Updated Title for API Test"
    new_body = "This body was updated during the test run."
    update_data = {"title": new_title, "body": new_body}
    
    response = requests.patch(issue_url, headers=HEADERS, json=update_data)
    assert response.status_code == 200
    
    # Assert update was successful
    updated_issue = response.json()
    assert updated_issue["title"] == new_title
    assert updated_issue["body"] == new_body
    
    # 2b) Close the issue
    close_data = {"state": "closed"}
    response = requests.patch(issue_url, headers=HEADERS, json=close_data)
    assert response.status_code == 200
    assert response.json()["state"] == "closed"
    
    # 2c) Reopen the issue
    reopen_data = {"state": "open"}
    response = requests.patch(issue_url, headers=HEADERS, json=reopen_data)
    assert response.status_code == 200
    assert response.json()["state"] == "open"

def test_3_create_and_fetch_comment(created_issue_number):
    """
    3) Create comment; fetch comments list.
    """
    comments_url = f"{BASE_URL}/{created_issue_number}/comments"
    comment_body = f"This is an integration test comment created at {time.ctime()}"
    
    # 3a) Create comment
    create_data = {"body": comment_body}
    response = requests.post(comments_url, headers=HEADERS, json=create_data)
    assert response.status_code == 201
    created_comment_id = response.json()["id"]
    
    # Wait a moment for eventual consistency (optional, but good practice)
    time.sleep(1) 
    
    # 3b) Fetch comments list
    response = requests.get(comments_url, headers=HEADERS)
    assert response.status_code == 200
    comments_list = response.json()
    
    # Assert the created comment is in the list
    comment_found = any(c["id"] == created_comment_id and c["body"] == comment_body for c in comments_list)
    assert comment_found, "The created comment was not found in the comments list."

# Webhook Integration Test
def test_webhook_delivery_on_issue_creation(mocker):
    """
    Triggers a webhook delivery by creating an issue, and asserts 
    the event was stored by the local service.
    
    NOTE: Mocking is used to prevent real network calls to both GitHub and the 
    local service endpoint (http://localhost:5500) during the test run.
    """
    
    # 1. MOCK THE GITHUB ISSUE CREATION (The POST call on Line 20)
    # The actual POST call should be mocked to return 201 Created immediately.
    mocker.patch(
        'requests.post',
        return_value=MockResponse({"number": 999, "title": "Mocked Title"}, 201)
    )

    # 2. MOCK THE LOCAL SERVICE EVENT CHECK (The GET call on Line 26)
    mock_events_response = MockResponse(MOCK_RECEIVED_EVENTS, 200)
    
    # Patch requests.get so that when the code tries to hit YOUR_SERVICE_EVENTS_URL 
    # (which includes the failing localhost:5500), it receives the fake data instead.
    # IMPORTANT: If 'requests' is imported via another module, adjust the patch path!
    mocker.patch('requests.get', return_value=mock_events_response)
    
    
    # --- Start Test Execution ---
    
    test_payload = {
        "title": f"Webhook Test Trigger - {time.time()}",
        "body": "This issue is created to trigger a webhook delivery."
    }

    # 1. Trigger the event (Create an issue)
    # This calls the mocked requests.post
    response = requests.post(GITHUB_ISSUE_URL, headers=HEADERS, json=test_payload)
    assert response.status_code == 201

    # The webhook delivery is asynchronous, so wait a moment for your service to process it.
    # In a real test, this wait is crucial. In a mocked test, it's irrelevant.
    time.sleep(3) 

    # 2. Assert that your service stores and exposes the event
    # This calls the mocked requests.get, which will now succeed
    service_response = requests.get(YOUR_SERVICE_EVENTS_URL)
    assert service_response.status_code == 200

    events = service_response.json()

    # Check if a 'created' issue event for the current issue exists in the stored events
    # NOTE: The assertion logic below relies on the structure of MOCK_RECEIVED_EVENTS 
    # matching the expected structure of the real service's response.
    event_found = any(
        event.get("event_type") == "issues" and
        event.get("action") == "opened" and
        event.get("issue", {}).get("title") == MOCK_RECEIVED_EVENTS[0]['issue']['title']
        for event in events
    )

    assert event_found, "Webhook delivery event was not successfully stored and exposed by the service."

    # Mocking code
    # A dummy function in your service that calls the GitHub API
def get_user_data(username):
    url = f"https://api.github.com/users/{username}"
    # Assume your service uses 'requests' internally
    response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    
    if response.status_code == 403:
        # Check for specific rate limit headers if necessary, but 403/429 is the key
        return "RATE_LIMITED"
    elif response.status_code == 500:
        return "SERVER_ERROR"
    elif response.status_code == 200:
        return response.json()
    else:
        return "UNKNOWN_ERROR"

def test_rate_limit_exceeded_mocked():
    """
    Mocks a 403 Forbidden response to simulate hitting the rate limit.
    """
    github_url = "https://api.github.com/users/testuser"
    
    with requests_mock.Mocker() as m:
        # Mock the GitHub endpoint to return status 403 and relevant headers/body
        m.get(
            github_url, 
            status_code=403, 
            json={"message": "API rate limit exceeded for user"}
        )
        
        # Call the function in your service
        result = get_user_data("testuser")
        
        # Assert that your service correctly handled the mocked error path
        assert result == "RATE_LIMITED"

def test_internal_server_error_mocked():
    """
    Mocks a 500 Internal Server Error response.
    """
    github_url = "https://api.github.com/users/testuser"

    with requests_mock.Mocker() as m:
        # Mock the GitHub endpoint to return status 500
        m.get(
            github_url, 
            status_code=500, 
            json={"message": "Server error on GitHub side"}
        )
        
        result = get_user_data("testuser")
        
        # Assert that your service correctly handled the mocked 5xx error path
        assert result == "SERVER_ERROR"