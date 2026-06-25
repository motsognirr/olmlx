"""Tests for the minimal operator web UI mounted at / (#373).

The UI is served via content negotiation: browsers (Accept: text/html) get the
single-page dashboard, while API/heartbeat clients (curl, the Ollama Go client)
continue to receive the plain-text "Ollama is running" response so existing
integrations keep working.
"""

import pytest


@pytest.mark.asyncio
async def test_root_serves_html_to_browsers(app_client):
    resp = await app_client.get("/", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "<title>olmlx" in body
    # The dashboard loads its script and stylesheet from the static mount.
    assert "/ui/app.js" in body
    assert "/ui/style.css" in body


@pytest.mark.asyncio
async def test_root_plain_text_for_api_clients(app_client):
    resp = await app_client.get("/", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert resp.text == "Ollama is running"
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_root_default_accept_is_plain_text(app_client):
    """Default httpx Accept is */* — heartbeat clients must keep the plain text."""
    resp = await app_client.get("/")
    assert resp.status_code == 200
    assert resp.text == "Ollama is running"


@pytest.mark.asyncio
async def test_head_root_ok(app_client):
    resp = await app_client.head("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_static_app_js_served(app_client):
    resp = await app_client.get("/ui/app.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_static_style_css_served(app_client):
    resp = await app_client.get("/ui/style.css")
    assert resp.status_code == 200
    assert "css" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_index_html_served_from_static_mount(app_client):
    resp = await app_client.get("/ui/index.html")
    assert resp.status_code == 200
    assert "<title>olmlx" in resp.text
