# Author: Zach Xie
# Contributor(s):

from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import JSONResponse
from .schemas import (
    CreateIssueRequest, UpdateIssueRequest, CreateCommentRequest, Issue, Comment, ErrorResponse, EventRecord
)
from .github_client import client
from .config import settings
from .utils import verify_signature, EVENT_HEADER, DELIVERY_HEADER
from .logging_utils import logger, RequestIdMiddleware
from datetime import datetime, timezone

app = FastAPI(title="GitHub Issues Gateway", version="1.0.0")
app.add_middleware(RequestIdMiddleware)

@app.on_event("startup")
async def _startup():
    logger.info("App started!")

    # Test: call GET /issues directly
    try:
        data, _ = await client.list_issues({"state": "open", "per_page": 1})
        if data:
            logger.info("startup_test_list_issues", issue_number=data[0]["number"], title=data[0]["title"])
        else:
            logger.info("startup_test_list_issues", message="No open issues found")
    except Exception as e:
        logger.error("startup_test_list_issues_failed", error=str(e))


# 1) POST /issues, create new issues
# Author: Zach Xie
@app.post("/issues", response_model=Issue, status_code=201, responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}})
async def create_issue(req: CreateIssueRequest, response: Response, request: Request):
    payload = req.model_dump(exclude_none=True)
    now = datetime.now(timezone.utc).isoformat()
    logger.debug("Calling POST /issues ...", timestamp=now)
    try:
        data, headers = await client.create_issue(payload)
        response.headers.update(headers)
        response.headers["Location"] = f"/issues/{data['number']}"

        # Log successful response
        issue_number = data['number']
        logger.info("Issue created!",
            timestamp=now,
            issue_number=issue_number,
        )
        return data
    except HTTPException as e:
        # Log failed response
        logger.error(f"Issues creation failed with {e.status_code}",
            timestamp=now,
        )
        if e.status_code == 401:
            return JSONResponse(status_code=401, content={"error": "unauthorized", "details": e.detail})
        raise

# 2) GET /issues, get all issues
# Author: Zach Xie
@app.get("/issues", response_model=list[Issue])
async def list_issues(state: str = "open", labels: str | None = None, page: int = 1, per_page: int = 30, response: Response = None):
    now = datetime.now(timezone.utc).isoformat()
    logger.debug("Calling GET /issues with param ...", 
        timestamp=now,
        state=state,
        page=page,
        per_page=per_page,
        labels=labels
    )
    if per_page > 100:
        raise HTTPException(status_code=400, detail={"error": "per_page must be <= 100"})
    params = {"state": state, "page": page, "per_page": per_page}
    if labels:
        params["labels"] = labels
    data, headers = await client.list_issues(params)
    if response:
        response.headers.update(headers)
    return data