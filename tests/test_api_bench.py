"""Tests for olmlx.bench.api_bench — adapter request shapes and stream parsing."""

from __future__ import annotations

import json

import pytest

from olmlx.bench.api_bench import (
    ADAPTERS,
    AnthropicMessagesAdapter,
    OllamaChatAdapter,
    OllamaGenerateAdapter,
    OpenAIChatAdapter,
    RunRecord,
    _estimate_tokens,
    _pick_apis,
    _pick_modes,
    _pick_prompts,
    summarize,
)
from olmlx.bench.prompts import BenchPrompt


@pytest.fixture
def simple_prompt() -> BenchPrompt:
    return BenchPrompt(
        name="t",
        category="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32,
    )


class TestOllamaChatAdapter:
    def test_build_request_shape(self, simple_prompt):
        path, body, headers = OllamaChatAdapter().build_request(
            simple_prompt, "qwen3:8b", 64, stream=True
        )
        assert path == "/api/chat"
        assert body["model"] == "qwen3:8b"
        assert body["stream"] is True
        assert body["messages"] == simple_prompt.messages
        assert body["options"]["num_predict"] == 64
        assert body["options"]["temperature"] == 0.0
        assert headers["Content-Type"] == "application/json"

    def test_parse_nonstream(self):
        resp = {
            "message": {"role": "assistant", "content": "hi there"},
            "prompt_eval_count": 5,
            "eval_count": 2,
            "prompt_eval_duration": 1_000_000,
            "eval_duration": 2_000_000,
        }
        m = OllamaChatAdapter().parse_nonstream(resp)
        assert m.text == "hi there"
        assert m.prompt_tokens == 5
        assert m.output_tokens == 2
        assert m.prompt_eval_duration_ns == 1_000_000
        assert m.eval_duration_ns == 2_000_000

    def test_iter_stream(self):
        lines = [
            json.dumps(
                {"message": {"role": "assistant", "content": "he"}, "done": False}
            ),
            json.dumps(
                {"message": {"role": "assistant", "content": "llo"}, "done": False}
            ),
            json.dumps(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 7,
                    "eval_count": 3,
                    "eval_duration": 500_000_000,
                    "prompt_eval_duration": 100_000_000,
                }
            ),
        ]
        events = list(OllamaChatAdapter().iter_stream(iter(lines)))
        texts = [e.text for e in events if e.text]
        assert texts == ["he", "llo"]
        done = [e for e in events if e.done]
        assert len(done) == 1
        assert done[0].output_tokens == 3
        assert done[0].prompt_tokens == 7
        assert done[0].eval_duration_ns == 500_000_000


class TestOllamaGenerateAdapter:
    def test_flatten_multi_turn(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        out = OllamaGenerateAdapter._flatten(msgs)
        assert "user: A" in out
        assert "assistant: B" in out
        assert out.endswith("user: C")

    def test_build_request(self, simple_prompt):
        path, body, _ = OllamaGenerateAdapter().build_request(
            simple_prompt, "m", 16, stream=False
        )
        assert path == "/api/generate"
        assert "prompt" in body
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 16

    def test_iter_stream(self):
        lines = [
            json.dumps({"response": "foo", "done": False}),
            json.dumps(
                {"response": "", "done": True, "eval_count": 1, "prompt_eval_count": 2}
            ),
        ]
        events = list(OllamaGenerateAdapter().iter_stream(iter(lines)))
        assert [e.text for e in events if e.text] == ["foo"]
        done = [e for e in events if e.done][0]
        assert done.output_tokens == 1
        assert done.prompt_tokens == 2


class TestOpenAIChatAdapter:
    def test_build_request(self, simple_prompt):
        path, body, _ = OpenAIChatAdapter().build_request(
            simple_prompt, "m", 32, stream=True
        )
        assert path == "/v1/chat/completions"
        assert body["stream"] is True
        assert body["max_tokens"] == 32
        assert body["messages"] == simple_prompt.messages

    def test_parse_nonstream(self):
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        m = OpenAIChatAdapter().parse_nonstream(resp)
        assert m.text == "ok"
        assert m.prompt_tokens == 3
        assert m.output_tokens == 4

    def test_iter_stream_terminates_on_done(self):
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "he"}}]}),
            "",
            "data: " + json.dumps({"choices": [{"delta": {"content": "llo"}}]}),
            "",
            "data: [DONE]",
            "",
        ]
        events = list(OpenAIChatAdapter().iter_stream(iter(lines)))
        assert [e.text for e in events if e.text] == ["he", "llo"]
        assert events[-1].done is True

    def test_iter_stream_skips_empty_deltas(self):
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}),
            "data: [DONE]",
        ]
        events = list(OpenAIChatAdapter().iter_stream(iter(lines)))
        texts = [e.text for e in events if e.text]
        assert texts == ["x"]


class TestAnthropicMessagesAdapter:
    def test_build_request(self, simple_prompt):
        path, body, headers = AnthropicMessagesAdapter().build_request(
            simple_prompt, "m", 64, stream=True
        )
        assert path == "/v1/messages"
        assert body["max_tokens"] == 64
        assert body["stream"] is True
        assert headers.get("anthropic-version")

    def test_parse_nonstream_concats_text_blocks(self):
        resp = {
            "content": [
                {"type": "text", "text": "he"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                {"type": "text", "text": "llo"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 9},
        }
        m = AnthropicMessagesAdapter().parse_nonstream(resp)
        assert m.text == "hello"
        assert m.prompt_tokens == 5
        assert m.output_tokens == 9

    def test_iter_stream_sse(self):
        def sse(event, data):
            return [f"event: {event}", "data: " + json.dumps(data), ""]

        lines: list[str] = []
        lines += sse(
            "message_start",
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 11, "output_tokens": 0}},
            },
        )
        lines += sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        lines += sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "he"},
            },
        )
        lines += sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "llo"},
            },
        )
        lines += sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        lines += sse(
            "message_delta",
            {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 22}},
        )
        lines += sse("message_stop", {"type": "message_stop"})

        events = list(AnthropicMessagesAdapter().iter_stream(iter(lines)))
        texts = [e.text for e in events if e.text]
        assert texts == ["he", "llo"]
        done = [e for e in events if e.done]
        assert len(done) == 1
        assert done[0].prompt_tokens == 11
        assert done[0].output_tokens == 22


class TestHelpers:
    def test_estimate_tokens_nonempty(self):
        assert _estimate_tokens("hello world foo") >= 1

    def test_estimate_tokens_empty(self):
        assert _estimate_tokens("") == 0

    def test_pick_apis_default(self):
        all_apis = _pick_apis(None)
        assert {a.name for a in all_apis} == set(ADAPTERS.keys())

    def test_pick_apis_subset(self):
        apis = _pick_apis("openai-chat,anthropic-messages")
        assert [a.name for a in apis] == ["openai-chat", "anthropic-messages"]

    def test_pick_apis_unknown(self):
        with pytest.raises(SystemExit):
            _pick_apis("bogus-api")

    def test_pick_prompts_all(self):
        assert len(_pick_prompts(None)) >= 6

    def test_pick_prompts_subset(self):
        chosen = _pick_prompts("factual,coding")
        assert [p.name for p in chosen] == ["factual", "coding"]

    def test_pick_prompts_unknown(self):
        with pytest.raises(SystemExit):
            _pick_prompts("nope")

    def test_pick_modes(self):
        assert _pick_modes("stream,nostream") == ["stream", "nostream"]

    def test_pick_modes_unknown(self):
        with pytest.raises(SystemExit):
            _pick_modes("blocking")


class TestSummarize:
    def _rec(
        self,
        api="openai-chat",
        mode="stream",
        model="m",
        prompt="p",
        ttft=10.0,
        tps=50.0,
        total=200.0,
        error=None,
    ):
        return RunRecord(
            api=api,
            mode=mode,
            model=model,
            prompt=prompt,
            run_index=0,
            ttft_ms=ttft,
            total_ms=total,
            prompt_tokens=10,
            output_tokens=20,
            tokens_per_sec=tps,
            tokens_estimated=False,
            error=error,
        )

    def test_groups_by_api_mode_model_prompt(self):
        recs = [
            self._rec(ttft=10, tps=40),
            self._rec(ttft=20, tps=60),
            self._rec(api="ollama-chat", ttft=5, tps=100),
        ]
        s = summarize(recs)
        assert len(s) == 2
        openai = next(r for r in s if r["api"] == "openai-chat")
        assert openai["runs"] == 2
        assert openai["ttft_p50_ms"] == 15.0
        assert openai["tps_p50"] == 50.0

    def test_excludes_errored(self):
        recs = [self._rec(), self._rec(error="boom")]
        s = summarize(recs)
        assert len(s) == 1
        assert s[0]["runs"] == 1
