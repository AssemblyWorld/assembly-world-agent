"""Opt-in real WebMCP integration; no model calls or synthetic browser tools.

AWA_BROWSER_EPISODE must point at an initial episode with movable objects.
AWA_MCP_COMMAND must point at the installed pinned Chrome DevTools MCP executable.
"""

import asyncio
import json
import os
import re

import pytest

from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.experiments.browser import DEFAULT_ENVIRONMENT, Browser
from assembly_world_agent.experiments.serving import EpisodeServer
from assembly_world_agent.experiments.transport import Client, images

pytestmark = pytest.mark.skipif(
    not os.environ.get("AWA_BROWSER_EPISODE"), reason="Explicit live browser opt-in required"
)


@pytest.mark.parametrize("headless", [False, True], ids=["headed", "headless"])
def test_real_webmcp_isolation_export_reload(tmp_path, headless):
    source = os.environ["AWA_BROWSER_EPISODE"]
    initial = read_episode(source)
    options = {
        "environment_url": os.environ.get("AWA_ENVIRONMENT_URL", DEFAULT_ENVIRONMENT),
        "mcp_command": os.environ.get("AWA_MCP_COMMAND", "chrome-devtools-mcp"),
        "headless": headless,
    }
    expected = {"episode_id": initial["manifest"]["id"], "initial_calls": len(initial["calls"])}

    async def sample(number, server):
        root = tmp_path / str(number)
        root.mkdir()
        browser = Browser(root, options)
        client = None
        try:
            await browser.start(server.add(source))
            print(f"Worker {number}: URL loaded", flush=True)
            user_agent = await browser.page.evaluate("navigator.userAgent")
            assert ("HeadlessChrome" in user_agent) == headless
            client = await Client().start(browser.mcp_command())
            pages = await client.request("tools/call", {"name": "list_pages", "arguments": {}})
            text = "\n".join(c.get("text", "") for c in pages["content"])
            page_id = int(re.search(r"(?m)^(\d+):", text)[1])

            async def invoke(name, arguments):
                result = await client.request(
                    "tools/call",
                    {
                        "name": "execute_webmcp_tool",
                        "arguments": {
                            "pageId": page_id,
                            "toolName": name,
                            "input": json.dumps(arguments),
                        },
                    },
                )
                assert not result.get("isError"), result
                return result

            discovery = await client.request(
                "tools/call", {"name": "list_webmcp_tools", "arguments": {"pageId": page_id}}
            )
            assert "capture_scene" in json.dumps(discovery)
            await invoke("start_episode", {})
            await invoke("get_scene", {})
            await invoke(
                "translate_objects",
                {"ids": [initial["manifest"]["objects"][0]["id"]], "delta": [number / 10, 0, 0]},
            )
            capture = await invoke("capture_scene", {})
            assert list(images(capture))
            final = root / "final.episode.zip"
            await browser.export(final, expected)
            print(f"Worker {number}: full episode exported", flush=True)
            saved = read_episode(final)
            assert [call["name"] for call in saved["calls"][-2:]] == [
                "translate_objects",
                "capture_scene",
            ]
            assert saved["calls"][-2]["arguments"]["delta"] == [number / 10, 0, 0]
            assert all(call["status"] == "completed" for call in saved["calls"][-2:])
            model_path = "world/" + initial["manifest"].get("model", {}).get("path", "model.xml")
            assert saved["files"][model_path] == initial["files"][model_path]
        finally:
            if client:
                await client.close()
            await browser.close()
        reload_root = root / "reload"
        reload_root.mkdir()
        reopened = Browser(reload_root, options)
        try:
            await reopened.start(server.add(final))
            roundtrip = root / "roundtrip.episode.zip"
            await reopened.export(roundtrip, {**expected, "initial_calls": len(saved["calls"])})
            assert read_episode(roundtrip)["calls"] == saved["calls"]
            print(f"Worker {number}: roundtrip verified", flush=True)
        finally:
            await reopened.close()

    async def run():
        with EpisodeServer(options["environment_url"]) as server:
            async with asyncio.timeout(300):
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(sample(1, server))
                    tasks.create_task(sample(2, server))

    asyncio.run(run())


def test_headless_rejects_failed_episode_url(tmp_path):
    async def run():
        options = {
            "environment_url": os.environ.get("AWA_ENVIRONMENT_URL", DEFAULT_ENVIRONMENT),
            "headless": True,
        }
        with EpisodeServer(options["environment_url"]) as server:
            browser = Browser(tmp_path, options)
            try:
                with pytest.raises(RuntimeError, match="Episode download failed: 404"):
                    await browser.start(server.add(tmp_path / "missing.episode.zip"))
                assert not browser.loaded
            finally:
                await browser.close()

    asyncio.run(run())
