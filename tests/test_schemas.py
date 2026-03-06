"""Tests for all schema models."""

from mlx_ollama.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTool,
    AnthropicToolInputSchema,
    AnthropicUsage,
)
from mlx_ollama.schemas.chat import ChatRequest, ChatResponse, Message, Tool, ToolCall, ToolCallFunction
from mlx_ollama.schemas.common import ModelOptions
from mlx_ollama.schemas.embed import EmbedRequest, EmbedResponse, EmbeddingsRequest, EmbeddingsResponse
from mlx_ollama.schemas.generate import GenerateRequest, GenerateResponse
from mlx_ollama.schemas.manage import CopyRequest, CreateRequest, DeleteRequest
from mlx_ollama.schemas.models import ModelDetails, ModelInfo, ShowRequest, ShowResponse, TagsResponse
from mlx_ollama.schemas.openai import (
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAICompletionChoice,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIModel,
    OpenAIModelList,
    OpenAIUsage,
)
from mlx_ollama.schemas.pull import PullRequest, PullResponse
from mlx_ollama.schemas.status import PsResponse, RunningModel, VersionResponse


class TestCommonSchemas:
    def test_model_options_defaults(self):
        opts = ModelOptions()
        assert opts.temperature is None
        assert opts.top_p is None
        assert opts.seed is None

    def test_model_options_extra_allowed(self):
        opts = ModelOptions(custom_param=42)
        assert opts.custom_param == 42


class TestGenerateSchemas:
    def test_generate_request_defaults(self):
        req = GenerateRequest(model="test")
        assert req.prompt == ""
        assert req.stream is True
        assert req.raw is False

    def test_generate_response(self):
        resp = GenerateResponse(
            model="test", created_at="now", response="hello", done=True,
        )
        assert resp.response == "hello"
        assert resp.done is True


class TestChatSchemas:
    def test_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.images is None

    def test_tool_call(self):
        tc = ToolCall(function=ToolCallFunction(name="f", arguments={"x": 1}))
        assert tc.function.name == "f"

    def test_chat_request_defaults(self):
        req = ChatRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )
        assert req.stream is True
        assert req.tools is None

    def test_chat_response(self):
        resp = ChatResponse(
            model="test",
            created_at="now",
            message=Message(role="assistant", content="hi"),
            done=True,
        )
        assert resp.done is True


class TestModelSchemas:
    def test_model_details_defaults(self):
        d = ModelDetails()
        assert d.format == "mlx"
        assert d.family == ""

    def test_model_info(self):
        info = ModelInfo(name="test:latest")
        assert info.size == 0
        assert info.digest == ""

    def test_tags_response(self):
        resp = TagsResponse(models=[ModelInfo(name="a"), ModelInfo(name="b")])
        assert len(resp.models) == 2

    def test_show_request(self):
        req = ShowRequest(model="test")
        assert req.verbose is False


class TestPullSchemas:
    def test_pull_request_defaults(self):
        req = PullRequest(model="test")
        assert req.stream is True
        assert req.insecure is False

    def test_pull_response(self):
        resp = PullResponse(status="downloading")
        assert resp.digest is None


class TestEmbedSchemas:
    def test_embed_request_string(self):
        req = EmbedRequest(model="test", input="hello")
        assert req.input == "hello"

    def test_embed_request_list(self):
        req = EmbedRequest(model="test", input=["a", "b"])
        assert len(req.input) == 2

    def test_embeddings_request(self):
        req = EmbeddingsRequest(model="test", prompt="hello")
        assert req.prompt == "hello"


class TestManageSchemas:
    def test_copy_request(self):
        req = CopyRequest(source="a", destination="b")
        assert req.source == "a"

    def test_delete_request(self):
        req = DeleteRequest(model="test")
        assert req.model == "test"

    def test_create_request_defaults(self):
        req = CreateRequest(model="test")
        assert req.modelfile is None
        assert req.stream is True


class TestStatusSchemas:
    def test_running_model(self):
        rm = RunningModel(name="test")
        assert rm.size == 0
        assert rm.size_vram == 0

    def test_ps_response(self):
        resp = PsResponse(models=[])
        assert resp.models == []

    def test_version_response(self):
        resp = VersionResponse(version="0.1.0")
        assert resp.version == "0.1.0"


class TestOpenAISchemas:
    def test_chat_message(self):
        msg = OpenAIChatMessage(role="user", content="hi")
        assert msg.tool_calls is None

    def test_chat_request_defaults(self):
        req = OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.n == 1
        assert req.frequency_penalty == 0.0

    def test_chat_response(self):
        resp = OpenAIChatResponse(
            id="test", created=0, model="m",
            choices=[OpenAIChoice(message=OpenAIChatMessage(role="assistant", content="hi"))],
        )
        assert resp.object == "chat.completion"

    def test_completion_request(self):
        req = OpenAICompletionRequest(model="test", prompt="hello")
        assert req.stream is False

    def test_completion_response(self):
        resp = OpenAICompletionResponse(
            id="test", created=0, model="m",
            choices=[OpenAICompletionChoice(text="world")],
        )
        assert resp.object == "text_completion"

    def test_model(self):
        m = OpenAIModel(id="test")
        assert m.object == "model"
        assert m.owned_by == "mlx-ollama"

    def test_model_list(self):
        ml = OpenAIModelList(data=[OpenAIModel(id="a"), OpenAIModel(id="b")])
        assert len(ml.data) == 2

    def test_embedding_request_string(self):
        req = OpenAIEmbeddingRequest(model="test", input="hello")
        assert req.encoding_format == "float"

    def test_embedding_response(self):
        resp = OpenAIEmbeddingResponse(
            data=[OpenAIEmbeddingData(embedding=[0.1, 0.2])],
            model="test",
        )
        assert resp.object == "list"

    def test_usage(self):
        u = OpenAIUsage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0

    def test_max_completion_tokens(self):
        req = OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            max_completion_tokens=1024,
        )
        assert req.max_completion_tokens == 1024


class TestAnthropicSchemas:
    def test_tool_input_schema(self):
        schema = AnthropicToolInputSchema(
            properties={"x": {"type": "string"}},
            required=["x"],
        )
        assert schema.type == "object"

    def test_tool_input_schema_extra(self):
        schema = AnthropicToolInputSchema(additionalProperties=False)
        assert schema.additionalProperties is False

    def test_tool(self):
        tool = AnthropicTool(
            name="get_weather",
            description="Get weather",
            input_schema=AnthropicToolInputSchema(),
        )
        assert tool.name == "get_weather"

    def test_content_block_text(self):
        block = AnthropicContentBlock(type="text", text="hello")
        assert block.text == "hello"

    def test_content_block_tool_use(self):
        block = AnthropicContentBlock(
            type="tool_use", id="toolu_123", name="func", input={"x": 1},
        )
        assert block.name == "func"

    def test_content_block_tool_result(self):
        block = AnthropicContentBlock(
            type="tool_result", tool_use_id="toolu_123", content="result",
        )
        assert block.tool_use_id == "toolu_123"

    def test_message_string_content(self):
        msg = AnthropicMessage(role="user", content="hello")
        assert msg.content == "hello"

    def test_message_blocks_content(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[AnthropicContentBlock(type="text", text="hi")],
        )
        assert len(msg.content) == 1

    def test_messages_request_defaults(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        assert req.max_tokens == 4096
        assert req.stream is False
        assert req.tools is None
        assert req.system is None

    def test_messages_request_with_system(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system="You are helpful.",
        )
        assert req.system == "You are helpful."

    def test_messages_response(self):
        resp = AnthropicMessagesResponse(
            id="msg_123",
            content=[AnthropicContentBlock(type="text", text="hi")],
            model="test",
            usage=AnthropicUsage(input_tokens=10, output_tokens=20),
        )
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.stop_reason == "end_turn"

    def test_usage(self):
        u = AnthropicUsage()
        assert u.input_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0
