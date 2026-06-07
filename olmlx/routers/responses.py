import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.grammar import GrammarSpec, parse_response_format
from olmlx.engine.inference import generate_chat
from olmlx.engine.responses_state import get_store
from olmlx.engine.tool_parser import (
    fill_missing_required_args,
    parse_model_output,
    resolve_tool_names,
)
from olmlx.routers.common import build_inference_options, resolve_openai_think
from olmlx.schemas.responses import ResponsesRequest, ResponsesResponse
from olmlx.utils.images import normalize_image_block

logger = logging.getLogger(__name__)

router = APIRouter()

_BUILTIN_TOOL_TYPES = {"web_search", "code_interpreter", "computer_use", "file_search"}


def _make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _make_reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex[:24]}"


def _make_fc_id() -> str:
    return f"fc_{uuid.uuid4().hex[:24]}"


def _make_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _message_item_to_engine(item: dict) -> dict:
    """Convert a Responses `message` input item to an engine message dict."""
    role = item["role"]
    content = item.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    texts: list[str] = []
    images: list[str] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "text", "output_text"):
            txt = part.get("text") or ""
            if txt:
                texts.append(txt)
        elif ptype in ("input_image", "image_url"):
            raw = part.get("image_url")
            if isinstance(raw, str):
                block = {"type": "image_url", "image_url": {"url": raw}}
            else:
                block = {"type": "image_url", "image_url": raw or {}}
            images.append(normalize_image_block(block))
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")
    msg: dict = {"role": role, "content": " ".join(texts)}
    if images:
        msg["images"] = images
    return msg


def _build_input_messages(input_data: str | list[dict]) -> list[dict]:
    """Translate a Responses `input` field into engine message dicts."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    messages: list[dict] = []
    for item in input_data:
        itype = item.get("type")
        role = item.get("role")
        if itype in (None, "message") and role is not None:
            messages.append(_message_item_to_engine(item))
        elif itype == "function_call":
            name = item.get("name")
            if not name:
                raise ValueError("function_call item missing 'name'")
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
        else:
            raise ValueError(f"unsupported input item type: {itype!r}")
    return messages


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert Responses function tools to engine OpenAI-nested format."""
    if not tools:
        return None
    converted: list[dict] = []
    for t in tools:
        ttype = t.get("type")
        if ttype != "function":
            if ttype in _BUILTIN_TOOL_TYPES:
                raise ValueError(
                    f"built-in tool {ttype!r} is not supported; only custom "
                    "'function' tools are available"
                )
            raise ValueError(f"unsupported tool type: {ttype!r}")
        name = t.get("name")
        if not name:
            raise ValueError("function tool missing 'name'")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
        )
    return converted


def _grammar_from_text_format(text_cfg: dict | None) -> GrammarSpec | None:
    """Map a Responses `text.format` config to a GrammarSpec (or None)."""
    fmt = (text_cfg or {}).get("format")
    if not fmt:
        return None
    ftype = fmt.get("type")
    if ftype in (None, "text"):
        return None
    if ftype == "json_object":
        return parse_response_format({"type": "json_object"})
    if ftype == "json_schema":
        return parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "schema"),
                    "schema": fmt.get("schema", {}),
                },
            }
        )
    raise ValueError(f"unsupported text.format type: {ftype!r}")


def _resolve_reasoning(reasoning: dict | None) -> bool | None:
    """Map Responses `reasoning.effort` to the engine enable_thinking flag."""
    effort = (reasoning or {}).get("effort")
    return resolve_openai_think(effort, None)
