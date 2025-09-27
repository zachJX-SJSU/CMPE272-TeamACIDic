# Author: Zach Xie
# Contributor(s):

from pydantic import BaseModel, Field
import os

class Settings(BaseModel):
    github_token: str = Field(..., alias="GITHUB_TOKEN")
    github_owner: str = Field(..., alias="GITHUB_OWNER")
    github_repo: str = Field(..., alias="GITHUB_REPO")
    webhook_secret: str = Field(..., alias="WEBHOOK_SECRET")
    port: int = Field(default=5000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_log_sample_rate: float = Field(default=1.0, alias="REQUEST_LOG_SAMPLE_RATE")

    model_config = dict(populate_by_name=True)

settings = Settings(**{k: v for k, v in os.environ.items() if k})