from pydantic import BaseModel


class CopyRequest(BaseModel):
    source: str
    destination: str


class DeleteRequest(BaseModel):
    model: str


class CreateRequest(BaseModel):
    model: str
    modelfile: str | None = None
    stream: bool = True
    path: str | None = None
    quantize: str | None = None
