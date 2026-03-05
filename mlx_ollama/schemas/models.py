from pydantic import BaseModel


class ModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "mlx"
    family: str = ""
    families: list[str] | None = None
    parameter_size: str = ""
    quantization_level: str = ""


class ModelInfo(BaseModel):
    name: str
    model: str = ""
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = ModelDetails()


class TagsResponse(BaseModel):
    models: list[ModelInfo]


class ShowRequest(BaseModel):
    model: str
    verbose: bool = False


class ShowResponse(BaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails = ModelDetails()
    model_info: dict = {}
    modified_at: str = ""
