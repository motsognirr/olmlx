import copy
import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDFormatter(logging.Formatter):
    """Formatter that prefixes log messages with request ID from ContextVar."""

    def format(self, record):
        request_id = request_id_var.get()
        if request_id:
            record = copy.copy(record)
            record.msg = f"[{request_id[:8]}] {record.getMessage()}"
            record.args = ()
        return super().format(record)
