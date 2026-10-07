"""Ollama Modelfile parsing for ``/api/create`` (#760).

Only the subset olmlx can honour is extracted: ``FROM``, ``SYSTEM`` and the
``PARAMETER`` keys that map onto per-model options. Instructions that would
change what the model *is* (``TEMPLATE``, ``ADAPTER``, ``MESSAGE``) are
reported so the caller can reject them instead of silently dropping them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from olmlx.engine.registry import VALID_OPTION_KEYS

logger = logging.getLogger(__name__)

#: Modelfile instructions olmlx cannot apply. ``LICENSE``/``REQUIRES`` are
#: metadata and are ignored rather than rejected.
UNSUPPORTED_INSTRUCTIONS = frozenset({"TEMPLATE", "ADAPTER", "MESSAGE"})
_IGNORED_INSTRUCTIONS = frozenset({"LICENSE", "REQUIRES"})

_INT_OPTIONS = frozenset({"top_k", "seed", "num_predict", "repeat_last_n"})


@dataclass
class Modelfile:
    from_model: str | None = None
    system: str | None = None
    #: Raw ``PARAMETER`` values (strings), ``stop`` accumulated as a list.
    parameters: dict[str, Any] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def parse_modelfile(text: str) -> Modelfile:
    """Parse a Modelfile. Raises ``ValueError`` on malformed input."""
    mf = Modelfile()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        command, _, rest = line.partition(" ")
        command = command.upper()
        rest = rest.strip()
        # Triple-quoted values may span lines: SYSTEM """ ... """
        if rest.startswith('"""'):
            body = rest[3:]
            if '"""' in body:
                value = body[: body.index('"""')]
            else:
                parts = [body]
                while i < len(lines):
                    nxt = lines[i]
                    i += 1
                    if '"""' in nxt:
                        parts.append(nxt[: nxt.index('"""')])
                        break
                    parts.append(nxt)
                else:
                    raise ValueError(f'unterminated """ in {command}')
                value = "\n".join(parts)
            value = value.strip("\n")
        else:
            value = _unquote(rest)

        if command == "FROM":
            mf.from_model = value
        elif command == "SYSTEM":
            mf.system = value
        elif command == "PARAMETER":
            key, _, raw = value.partition(" ")
            raw = _unquote(raw.strip())
            if not key or not raw:
                raise ValueError(f"malformed PARAMETER line: {line!r}")
            if key == "stop":
                mf.parameters.setdefault("stop", []).append(raw)
            else:
                mf.parameters[key] = raw
        elif command in UNSUPPORTED_INSTRUCTIONS:
            mf.unsupported.append(command)
        elif command in _IGNORED_INSTRUCTIONS:
            continue
        else:
            raise ValueError(f"unknown Modelfile instruction: {command}")
    return mf


def coerce_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Map Ollama parameters onto olmlx per-model options.

    Keys with no olmlx equivalent (``num_ctx``, ``mirostat``, ...) are
    dropped with a log line, matching how request ``options`` treat them.
    String values (from a Modelfile) are converted to the option's type;
    a value that doesn't convert raises ``ValueError``.
    """
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key not in VALID_OPTION_KEYS:
            logger.info("Ignoring unsupported model parameter %r", key)
            continue
        if value is None:
            continue
        if key == "stop":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                raise ValueError(
                    "parameter 'stop' must be a string or a list of strings"
                )
            out[key] = list(value)
            continue
        if isinstance(value, str):
            try:
                value = int(value) if key in _INT_OPTIONS else float(value)
            except ValueError:
                raise ValueError(
                    f"invalid value for parameter {key!r}: {value!r}"
                ) from None
        out[key] = value
    return out
