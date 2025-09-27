# Author: Zach Xie
# Contributor(s):

import structlog
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.EventRenamer("message"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# Adding a middleware that assign each request with
# an id if not provided, helps with logging/tracing

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response