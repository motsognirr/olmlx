"""Streaming adapter for speculative decoding (Flash compatibility shim).

Re-exports from the base module for backward compatibility.
"""

from olmlx.engine.speculative_stream import (
    TokenizerProtocol,
    async_speculative_stream,
    speculative_stream_generate,
)

__all__ = [
    "TokenizerProtocol",
    "async_speculative_stream",
    "speculative_stream_generate",
]
