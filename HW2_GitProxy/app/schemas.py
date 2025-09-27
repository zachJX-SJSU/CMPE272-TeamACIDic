# Author: Zach Xie
# Contributor(s):

from typing import List, Optional
from pydantic import BaseModel, Field

class CreateIssueRequest(BaseModel):
    title: str
    body: Optional[str] = None
    labels: Optional[List[str]] = None

class UpdateIssueRequest(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    state: Optional[str] = Field(None, pattern="^(open|closed)$")

class CreateCommentRequest(BaseModel):
    body: str

class Label(BaseModel):
    name: str

class Issue(BaseModel):
    number: int
    html_url: str
    state: str
    title: str
    body: str | None = None
    labels: List[Label] = []
    created_at: str
    updated_at: str

class Comment(BaseModel):
    id: int
    body: str
    user: dict
    created_at: str
    html_url: str

class ErrorResponse(BaseModel):
    error: str
    details: dict | None = None

class EventRecord(BaseModel):
    id: str
    event: str
    action: str | None = None
    issue_number: int | None = None
    timestamp: str