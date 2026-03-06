from pydantic import BaseModel


class ModelOptions(BaseModel, extra="allow"):
    """Ollama model options / parameters."""

    num_keep: int | None = None
    seed: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    tfs_z: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = None
    temperature: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    penalize_newline: bool | None = None
    stop: list[str] | None = None
    numa: bool | None = None
    num_ctx: int | None = None
    num_batch: int | None = None
    num_gpu: int | None = None
    main_gpu: int | None = None
    low_vram: bool | None = None
    vocab_only: bool | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    num_thread: int | None = None
