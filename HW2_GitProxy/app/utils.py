# Author: Zach Xie
# Contributor(s):

import hmac, hashlib
from fastapi import HTTPException
from .config import settings

SIG_HEADER = "X-Hub-Signature-256"
EVENT_HEADER = "X-GitHub-Event"
DELIVERY_HEADER = "X-GitHub-Delivery"

def compute_signature(body: bytes) -> str:
    mac = hmac.new(settings.webhook_secret.encode(), body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"

def verify_signature(body: bytes, signature: str | None):
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature")
    expected = compute_signature(body)
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

def parse_pagination_headers(r):
    keys = ["link", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after"]
    return {k.capitalize() if k=="link" else k: v for k, v in r.headers.items() if k.lower() in keys}