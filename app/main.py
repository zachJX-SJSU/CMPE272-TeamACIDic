import hmac, hashlib, json, os, time
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
import aiosqlite
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))

app = FastAPI(title="GitHub Issues Gateway")

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

def verify_signature(secret: str, body: bytes, signature_header: Optional[str]) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    received = signature_header.split("=", 1)[1]
    mac = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()
    return hmac.compare_digest(received, expected)

@app.post("/webhook")
async def webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256"),
    x_github_event: Optional[str] = Header(default=None, alias="X-GitHub-Event"),
    x_github_delivery: Optional[str] = Header(default=None, alias="X-GitHub-Delivery"),
):
    raw = await request.body()
    if not verify_signature(WEBHOOK_SECRET, raw, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event not in ("issues", "issue_comment", "ping"):
        raise HTTPException(status_code=400, detail=f"Unsupported event: {x_github_event}")

    payload = json.loads(raw.decode("utf-8") or "{}")
    action = payload.get("action")
    issue_number = payload.get("issue", {}).get("number") if isinstance(payload.get("issue"), dict) else None

    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO events (id, event, action, issue_number, timestamp) VALUES (?, ?, ?, ?, ?)",
                (x_github_delivery, x_github_event, action, issue_number, now),
            )
            await db.commit()
        except aiosqlite.IntegrityError:
            pass  # duplicate delivery id → idempotent

    return Response(status_code=204)

class EventOut(BaseModel):
    id: str
    event: str
    action: Optional[str]
    issue_number: Optional[int]
    timestamp: int

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

@app.get("/healthz")
async def healthz():
    return {"ok": True}


