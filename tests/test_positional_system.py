"""System/developer turns after the first user turn (#740).

Qwen3.5/3.6 templates ``raise_exception("System message must be at the
beginning.")`` on any non-leading system turn. olmlx probes the template for
that constraint and folds late system turns into the leading one; any other
template ``raise_exception`` is a client-shape problem → 400, not 500.
"""

import pytest

from olmlx.engine.chat_templating import (
    ChatTemplateRejectedError,
    _apply_chat_template,
    _apply_chat_template_vlm,
    _fold_system_messages_to_front,
)
from olmlx.engine.template_caps import TemplateCaps, detect_caps

# Trimmed from the real Qwen3.6 chat_template.jinja.
STRICT_TEMPLATE = """
{%- for message in messages %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
        {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role in ["user", "assistant"] %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""

# Qwen3/Llama-style: renders a system turn wherever it appears.
POSITIONAL_TEMPLATE = """
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""

# Rejects every system turn, even a leading one — folding can't help.
NO_SYSTEM_TEMPLATE = """
{%- for message in messages %}
    {%- if message.role == "system" %}
        {{- raise_exception('System role not supported') }}
    {%- endif %}
    {{- message.content }}
{%- endfor %}
"""


class _JinjaTokenizer:
    """Minimal tokenizer that renders its chat_template like transformers."""

    def __init__(self, template: str):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        from transformers.utils.chat_template_utils import _compile_jinja_template

        return _compile_jinja_template(self.chat_template).render(
            messages=messages, **kwargs
        )


class TestDetectCaps:
    def test_strict_template_rejects_positional_system(self):
        caps = detect_caps(_JinjaTokenizer(STRICT_TEMPLATE))
        assert caps.rejects_positional_system is True

    def test_positional_template_keeps_positional_system(self):
        caps = detect_caps(_JinjaTokenizer(POSITIONAL_TEMPLATE))
        assert caps.rejects_positional_system is False

    def test_template_rejecting_all_system_is_not_flagged(self):
        # The baseline (leading system) render fails too, so folding wouldn't
        # help — don't flag it; the 400 fallback covers it.
        caps = detect_caps(_JinjaTokenizer(NO_SYSTEM_TEMPLATE))
        assert caps.rejects_positional_system is False

    def test_default_is_false(self):
        assert TemplateCaps().rejects_positional_system is False

    def test_real_qwen36_template(self):
        from pathlib import Path

        path = (
            Path.home()
            / ".olmlx/models/mlx-community_Qwen3.6-35B-A3B-4bit/chat_template.jinja"
        )
        if not path.exists():
            pytest.skip("Qwen3.6 template not in the local model store")
        caps = detect_caps(_JinjaTokenizer(path.read_text()))
        assert caps.rejects_positional_system is True


class TestFoldSystemMessages:
    def test_late_system_folded_into_leading(self):
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "r1"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u2"},
        ]
        assert _fold_system_messages_to_front(messages) == [
            {"role": "system", "content": "a\n\nb"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "u2"},
        ]

    def test_late_system_without_leading_becomes_leading(self):
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u2"},
        ]
        assert _fold_system_messages_to_front(messages) == [
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]

    def test_only_leading_system_unchanged(self):
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u1"},
        ]
        assert _fold_system_messages_to_front(messages) is messages

    def test_does_not_mutate_input(self):
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "b"},
        ]
        _fold_system_messages_to_front(messages)
        assert messages[0] == {"role": "system", "content": "a"}
        assert len(messages) == 3

    def test_folded_messages_render_on_strict_template(self):
        tok = _JinjaTokenizer(STRICT_TEMPLATE)
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "u2"},
        ]
        prompt = tok.apply_chat_template(
            _fold_system_messages_to_front(messages), add_generation_prompt=True
        )
        assert prompt.startswith("<|im_start|>system\nbe terse")


class TestTemplateRejectionIs400:
    def test_text_template_raise_is_value_error(self):
        tok = _JinjaTokenizer(NO_SYSTEM_TEMPLATE)
        with pytest.raises(
            ChatTemplateRejectedError, match="System role not supported"
        ):
            _apply_chat_template(tok, [{"role": "system", "content": "x"}])

    def test_rejection_is_value_error_and_runtime_error(self):
        # ValueError → the app's 400 handler; RuntimeError keeps existing
        # ``except RuntimeError`` fallbacks (/api/generate raw-prompt) working.
        assert issubclass(ChatTemplateRejectedError, ValueError)
        assert issubclass(ChatTemplateRejectedError, RuntimeError)

    def test_text_template_raise_with_tools_is_value_error(self):
        tok = _JinjaTokenizer(NO_SYSTEM_TEMPLATE)
        caps = TemplateCaps(supports_tools=True)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with pytest.raises(ChatTemplateRejectedError):
            _apply_chat_template(tok, [{"role": "system", "content": "x"}], tools, caps)

    def test_vlm_tools_template_raise_is_value_error(self):
        tok = _JinjaTokenizer(NO_SYSTEM_TEMPLATE)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with pytest.raises(ChatTemplateRejectedError):
            _apply_chat_template_vlm(
                tok, None, [{"role": "system", "content": "x"}], tools=tools
            )

    def test_non_template_errors_stay_runtime_error(self):
        class Boom:
            chat_template = ""

            def apply_chat_template(self, *a, **k):
                raise KeyError("oops")

        with pytest.raises(RuntimeError) as ei:
            _apply_chat_template(Boom(), [{"role": "user", "content": "x"}])
        assert not isinstance(ei.value, ValueError)


class TestFoldKeepsMetadata:
    def test_late_system_images_and_audio_carried_over(self):
        messages = [
            {"role": "system", "content": "a", "images": ["i1"]},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "b", "images": ["i2"], "audio": ["a1"]},
        ]
        lead = _fold_system_messages_to_front(messages)[0]
        assert lead["images"] == ["i1", "i2"]
        assert lead["audio"] == ["a1"]

    def test_name_kept_only_when_all_agree(self):
        same = [
            {"role": "system", "content": "a", "name": "x"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b", "name": "x"},
        ]
        assert _fold_system_messages_to_front(same)[0]["name"] == "x"
        differ = [
            {"role": "system", "content": "a", "name": "x"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b", "name": "y"},
        ]
        assert "name" not in _fold_system_messages_to_front(differ)[0]


class TestReviewFollowups:
    def test_probe_uses_inner_tokenizer_of_processor(self):
        class Processor:
            # Processor without its own apply_chat_template; the template and
            # the renderer live on the wrapped tokenizer (VLM layout).
            def __init__(self):
                self.tokenizer = _JinjaTokenizer(STRICT_TEMPLATE)

        caps = detect_caps(Processor())
        assert caps.rejects_positional_system is True

    def test_agreed_name_copied_onto_new_lead(self):
        messages = [
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b", "name": "x"},
        ]
        assert _fold_system_messages_to_front(messages)[0]["name"] == "x"


class TestProbeOnlyCountsDeliberateRejection:
    def test_late_render_failing_for_other_reasons_is_not_flagged(self):
        # The late render fails with a template *bug* (undefined attribute
        # access), not a raise_exception — must not trigger the fold.
        tpl = """
{%- for message in messages %}
    {%- if message.role == "system" and not loop.first %}
        {{- message.missing.attr }}
    {%- endif %}
    {{- message.content }}
{%- endfor %}
{%- if false %}{{ raise_exception('unused') }}{% endif %}
"""
        caps = detect_caps(_JinjaTokenizer(tpl))
        assert caps.rejects_positional_system is False
