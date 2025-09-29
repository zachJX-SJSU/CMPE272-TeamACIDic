"""
Custom exception classes for the application.
These exceptions are raised internally by the GitHubClient 
and mapped to appropriate HTTP responses in main.py.
"""

class GitHubClientError(Exception):
    """Base exception for all errors raised by the GitHubClient."""
    
    # We add attributes like status_code and detail so the exception handler 
    # (or main.py) knows how to format the HTTP response.
    def __init__(self, status_code: int, detail: str = "An unexpected error occurred."):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")

class NotFoundError(GitHubClientError):
    """Raised when a resource is not found (HTTP 404)."""
    def __init__(self, detail: str = "Resource not found."):
        super().__init__(status_code=404, detail=detail)

class AuthError(GitHubClientError):
    """Raised when authentication fails (HTTP 401 or 403)."""
    def __init__(self, detail: str = "Authentication or permissions failed."):
        super().__init__(status_code=401, detail=detail)

class BadRequestError(GitHubClientError):
    """Raised when input is invalid or a request is malformed (HTTP 400 or 422)."""
    def __init__(self, detail: str = "Bad Request or Invalid Input."):
        super().__init__(status_code=400, detail=detail)