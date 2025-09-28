# Author: Zach Xie
# Contributor(s): Archana Shivashankar

import httpx
from fastapi import HTTPException
from .config import settings 
from .utils import parse_pagination_headers 

BASE = "https://api.github.com"

class NotFoundError(Exception): pass
class AuthError(Exception): pass
class BadRequestError(Exception): pass

class GitHubClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=BASE,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {settings.github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )
        self.owner = settings.github_owner
        self.repo = settings.github_repo

    async def _handle_rate_limit(self, r: httpx.Response):
        if r.status_code in (403, 429):
            retry_after = r.headers.get("Retry-After")
            detail = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"message": r.text}
            raise HTTPException(status_code=503, detail={"error": "rate_limited", "retry_after": retry_after, "github": detail})

    async def create_issue(self, payload: dict):
        r = await self._client.post(f"/repos/{self.owner}/{self.repo}/issues", json=payload)
        await self._handle_rate_limit(r)
        if r.status_code == 422:
            raise BadRequestError("Validation failed due to bad request data.")
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
        raise HTTPException(status_code=r.status_code, detail=r.json())

    async def list_issues(self, params: dict):
        r = await self._client.get(f"/repos/{self.owner}/{self.repo}/issues", params=params)
        await self._handle_rate_limit(r)
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
        raise HTTPException(status_code=r.status_code, detail=r.json())

    async def get_issue(self, number: int):
        r = await self._client.get(f"/repos/{self.owner}/{self.repo}/issues/{number}")
        await self._handle_rate_limit(r)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail={"error": "not_found"})
        if r.status_code == 422:
            raise BadRequestError("Validation failed due to bad request data.")
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
        raise HTTPException(status_code=r.status_code, detail=r.json())

    async def update_issue(self, number: int, payload: dict):
        r = await self._client.patch(f"/repos/{self.owner}/{self.repo}/issues/{number}", json=payload)
        await self._handle_rate_limit(r)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail={"error": "not_found"})
        if r.status_code == 422:
            raise BadRequestError("Validation failed due to bad request data.")
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
        raise HTTPException(status_code=r.status_code, detail=r.json())

    async def create_comment(self, number: int, payload: dict):
        r = await self._client.post(f"/repos/{self.owner}/{self.repo}/issues/{number}/comments", json=payload)
        await self._handle_rate_limit(r)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail={"error": "not_found"})
        if r.status_code == 422:
            raise BadRequestError("Validation failed due to bad request data.")
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
        raise HTTPException(status_code=r.status_code, detail=r.json())

    async def close(self):
        await self._client.aclose()

    async def get_issue(self, number: int):
        r = await self._client.get(f"/repos/{self.owner}/{self.repo}/issues/{number}")
        await self._handle_rate_limit(r)
        
        # 1. Handle 404: Not Found
        if r.status_code == 404:
            raise NotFoundError("Resource not found.")
            
        # 2. Handle 422: Unprocessable Entity / Validation Failure
        if r.status_code == 422:
            # The message must align with the test assertion ("validation failed")
            raise BadRequestError("Validation failed due to bad request data.")

        # 3. Handle 401/403: Authorization Errors
        if r.status_code in (401, 403):
            raise AuthError("Authentication failed.")
            
        if r.is_success:
            return r.json(), parse_pagination_headers(r)
            
        # Fallback for other errors (e.g., 500, other unhandled 4xx)
        raise HTTPException(status_code=r.status_code, detail=r.json())

client = GitHubClient()