from pydantic import BaseModel


class PullRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True


class PullResponse(BaseModel):
    status: str
    digest: str | None = None
    total: int | None = None
    completed: int | None = None
