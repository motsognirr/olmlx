import pytest

from olmlx.utils.images import normalize_image_block


def test_openai_image_url_passthrough():
    block = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    assert normalize_image_block(block) == "http://x/y.png"


def test_openai_image_url_data_uri_passthrough():
    uri = "data:image/png;base64,AAAA"
    block = {"type": "image_url", "image_url": {"url": uri}}
    assert normalize_image_block(block) == uri


def test_anthropic_base64_builds_data_uri():
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QQ=="},
    }
    assert normalize_image_block(block) == "data:image/jpeg;base64,QQ=="


def test_anthropic_base64_defaults_media_type():
    block = {"type": "image", "source": {"type": "base64", "data": "QQ=="}}
    assert normalize_image_block(block) == "data:image/png;base64,QQ=="


def test_anthropic_url_source():
    block = {"type": "image", "source": {"type": "url", "url": "http://x/z.png"}}
    assert normalize_image_block(block) == "http://x/z.png"


def test_missing_url_raises():
    with pytest.raises(ValueError, match="image_url"):
        normalize_image_block({"type": "image_url", "image_url": {}})


def test_unsupported_source_type_raises():
    block = {"type": "image", "source": {"type": "file", "id": "abc"}}
    with pytest.raises(ValueError, match="unsupported image source"):
        normalize_image_block(block)


def test_not_an_image_block_raises():
    with pytest.raises(ValueError, match="not an image block"):
        normalize_image_block({"type": "text", "text": "hi"})
