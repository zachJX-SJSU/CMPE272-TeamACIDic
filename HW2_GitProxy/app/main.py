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
