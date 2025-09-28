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

#for webhooks: Pratham Rajesh
import os,hmac,hashlib,json,time
from typing import Optional, List
import aiosqlite
from pydantic import BaseModel

app = FastAPI(title="GitHub Issues Gateway", version="1.0.0")

#DB init block:Pratham Rajesh
@app.get("/healthz")
async def healthz():
    return {"ok": True}

DB_PATH = "events.db"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY,
              event TEXT NOT NULL,
              action TEXT,
              issue_number INTEGER,
              timestamp INTEGER NOT NULL
            )
        """)
        await db.commit()

@app.on_event("startup")
async def on_startup():
    await init_db()
#DB init end
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

# 2) GET /issues, get issues
# Author: Zach Xie
@app.get("/issues", response_model=list[Issue])
async def list_issues(state: str = "open", labels: str | None = None, page: int = 1, per_page: int = 30, response: Response = None):
    now = datetime.now(timezone.utc).isoformat()
    logger.debug("Calling GET /issues ...", 
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


# 3) GET /issues/{number}
# Author: Archana Shivashankar 
@app.get("/issues/{number}", response_model=Issue, responses={404: {"model": ErrorResponse}})
async def get_issue(number: int, response: Response):
   """
   Retrive a single issue form the Github repository by using its issue number.
   """
   #1. Log the incoming request
   now = datetime.now(timezone.utc).isoformat()
   logger.debug("Calling GET /issues/{number} ...",
       timestamp=now,
       issue_number=number
   )
   #2. Call the Github API to get the issue details
   try:
       data, headers = await client.get_issue(number)
   except Exception as e:
       raise HTTPException(status_code=500, detail={"error": f"Failed to fetch issue{number}", "details": str(e)})
   #3. update the response headers
   response.headers.update(headers)
   #4. return the issue details
   logger.info("Successfull got issue",
           timestamp=now,
           data=data
   )
   return data

# 4) PATCH /issues/{number}
# Author: Archana Shivashankar 
@app.patch("/issues/{number}", response_model=Issue, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def update_issue(number: int, update_data: UpdateIssueRequest, response: Response):
   """
   Update an existing issue in the Github repository by using its issue number.
   """
   #1. Log the incoming request
   now = datetime.now(timezone.utc).isoformat()
   logger.debug("Calling PATCH /issues/{number} ...",
       timestamp=now,
       issue_number=number,
       data=update_data.model_dump()
   )
   #2. Prepare the data to send to Github API
   data_to_send = update_data.model_dump(exclude_unset=True)

   #3. Call the Github API to update the issue details
   try:
       data, headers = await client.update_issue(number, data_to_send)
   except Exception as e:
       raise HTTPException(status_code=500, detail={"error": f"Failed to update issue{number}", "details": str(e)})
   #4. update the response headers
   response.headers.update(headers)
   logger.info("Successfull updated issue",
           timestamp=now,
           issue_numer=number,
           data=data
   )
   #5. return the updated issue details
   return data


# 5) POST /issues/{number}/comments
# Author: Preetam
@app.post(
    "/issues/{number}/comments",
    response_model=Comment,
    status_code=201,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def create_comment(number: int, req: CreateCommentRequest, response: Response):
    """
    Add a new comment to an existing GitHub issue.
    Request body: { "body": "your comment text" }
    """
    now = datetime.now(timezone.utc).isoformat()
    logger.debug("Calling POST /issues/{number}/comments ...",
        timestamp=now,
        issue_number=number,
        comment=req.body
    )

    try:
        payload = req.model_dump(exclude_none=True)
        data, headers = await client.create_comment(number, payload)
        response.headers.update(headers)

        logger.info("Comment created!",
            timestamp=now,
            issue_number=number,
            comment_id=data.get("id")
        )
        return data

    except HTTPException as e:
        logger.error(f"Comment creation failed with {e.status_code}",
            timestamp=now,
            issue_number=number
        )
        if e.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "details": e.detail}
            )
        raise

# 6) POST /webhook , handles webhook events from Github

# >>> PRATHAM WEBHOOK START >>>
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

class EventOut(BaseModel):
    id: str
    event: str
    action: Optional[str]
    issue_number: Optional[int]
    timestamp: int

@app.post("/webhook")
async def webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256"),
    x_github_event: Optional[str] = Header(default=None, alias="X-GitHub-Event"),
    x_github_delivery: Optional[str] = Header(default=None, alias="X-GitHub-Delivery"),
):
    raw = await request.body()

    # HMAC verify (uses utils.verify_signature imported near top)
    verify_signature(raw, x_hub_signature_256)

    # Support only these events
    if x_github_event not in ("issues", "issue_comment", "ping"):
        raise HTTPException(status_code=400, detail=f"Unsupported event: {x_github_event}")

    # Robust payload parsing: JSON or x-www-form-urlencoded (payload=<json>)
    content_type = (request.headers.get("content-type") or "").lower()
    body_text = raw.decode("utf-8") if raw else ""

    if "application/json" in content_type:
        body_json_text = body_text or "{}"
    elif "application/x-www-form-urlencoded" in content_type:
        import urllib.parse
        form = urllib.parse.parse_qs(body_text)
        body_json_text = form.get("payload", ["{}"])[0]
    else:
        body_json_text = body_text or "{}"

    try:
        payload = json.loads(body_json_text)
    except json.JSONDecodeError:
        # Acknowledge malformed/empty body so GitHub won't retry forever
        return Response(status_code=204)

    action = payload.get("action")
    issue = payload.get("issue") if isinstance(payload.get("issue"), dict) else None
    issue_number = issue.get("number") if isinstance(issue, dict) else None

    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO events (id, event, action, issue_number, timestamp) VALUES (?, ?, ?, ?, ?)",
                (x_github_delivery, x_github_event, action, issue_number, now),
            )
            await db.commit()
        except aiosqlite.IntegrityError:
            # duplicate delivery id -> ignore (idempotent)
            pass

    return Response(status_code=204)

@app.get("/events", response_model=List[EventOut])
async def get_events(n: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, event, action, issue_number, timestamp FROM events ORDER BY timestamp DESC LIMIT ?",
            (n,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [EventOut(**dict(r)) for r in rows]
# <<< PRATHAM WEBHOOK END <<<


# 7) GET /events , requests come from frontend (optional)
