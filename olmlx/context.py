import copy
import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# API surface for the current request: "ollama" | "openai" | "anthropic" |
# "audio". Set by MetricsMiddleware; read by engine inference instrumentation.
surface_var: ContextVar[str] = ContextVar("surface", default="unknown")


class RequestIDFormatter(logging.Formatter):
    """Formatter that prefixes log messages with request ID from ContextVar."""

    def format(self, record):
        request_id = request_id_var.get()
        if request_id:
            record = copy.copy(record)
            record.msg = f"[{request_id[:8]}] {record.getMessage()}"
            record.args = ()
        return super().format(record)
