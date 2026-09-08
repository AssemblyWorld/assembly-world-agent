"""Dedicated Chrome lifecycle, episode URL loading and public UI export."""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path
from urllib.parse import urlsplit

from ..episode_io import read_episode
from ..episodes import sha256
from .serving import environment_url

MCP_VERSION = "1.8.0"
DEFAULT_ENVIRONMENT = "https://3dwebagent.davidz.cn/"


def chrome_path(explicit=None):
    candidates = [
        explicit,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("Chrome not found; supply --chrome-path")


async def terminate(process):
    if process is None:
        return
    # Kill the owned process group even when its leader exited, to reap MCP children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


class Browser:
    def __init__(self, root, options):
        self.root, self.options = Path(root), options
        self.process = self.playwright = self.browser = None
        self.page = None
        self.loaded = False

    async def start(self, episode_url=None):
        from playwright.async_api import async_playwright

        profile = self.root / "chrome"
        profile.mkdir()
        self.process = await asyncio.create_subprocess_exec(
            chrome_path(self.options.get("chrome_path")),
            f"--user-data-dir={profile}",
            "--remote-debugging-port=0",
            "--remote-debugging-address=127.0.0.1",
            "--no-first-run",
            "--no-default-browser-check",
            "--enable-features=WebMCP",
            "--enable-blink-features=WebMCP,WebMCPTesting",
            *(["--headless"] if self.options.get("headless", False) else []),
            "about:blank",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        async with asyncio.timeout(30):
            while not (profile / "DevToolsActivePort").exists():
                if self.process.returncode is not None:
                    raise RuntimeError("Chrome exited during startup")
                await asyncio.sleep(0.1)
        port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        self.url = f"http://127.0.0.1:{port}"
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(self.url)
        context = self.browser.contexts[0]
        if episode_url:
            environment = urlsplit(self.options["environment_url"])
            await context.grant_permissions(
                ["local-network-access"], origin=f"{environment.scheme}://{environment.netloc}"
            )
        self.page = context.pages[0] if context.pages else await context.new_page()
        self.page.set_default_timeout(60000)
        self.cdp = await context.new_cdp_session(self.page)
        self.registered = set()
        self.cdp.on("WebMCP.toolsAdded", self._tools)
        await self.cdp.send("WebMCP.enable")
        target = self.options["environment_url"]
        if episode_url:
            target = environment_url(target, episode_url)
        await self.page.goto(target, wait_until="domcontentloaded")
        async with asyncio.timeout(90):
            while not self.registered:
                await asyncio.sleep(0.1)
        errors = [text.strip() for text in await self.page.get_by_role("alert").all_text_contents()]
        if any(errors):
            raise RuntimeError(f"Episode page failed: {'; '.join(filter(None, errors))}")
        self.loaded = episode_url is not None
        return self

    def _tools(self, event):
        self.registered.update(t["name"] for t in event.get("tools", []))

    def mcp_command(self):
        return [
            self.options["mcp_command"],
            "--browserUrl",
            self.url,
            "--categoryExperimentalWebmcp",
            "--no-usage-statistics",
            "--no-performance-crux",
        ]

    async def export(self, target, expected):
        if not self.loaded:
            raise RuntimeError("Input episode was not loaded")
        target = Path(target)
        temporary = self.root / "export.episode.zip"
        async with asyncio.timeout(60):
            async with self.page.expect_download() as pending:
                await self.page.get_by_text("File", exact=True).click()
                await self.page.get_by_role(
                    "button", name="Export full episode…", exact=True
                ).click()
            await (await pending.value).save_as(temporary)
        episode = await asyncio.to_thread(read_episode, temporary)
        if episode["manifest"]["id"] != expected["episode_id"]:
            raise ValueError("Exported episode identity differs from input")
        if len(episode["calls"]) < expected["initial_calls"]:
            raise ValueError("Export lost input call history")
        checksum = sha256(temporary.read_bytes())
        # Use a same-filesystem temporary file for the atomic replacement.
        staging = target.with_suffix(".tmp")
        try:
            shutil.copyfile(temporary, staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        return {"status": "saved", "sha256": checksum, "calls": len(episode["calls"])}

    async def close(self):
        try:
            if self.browser:
                await asyncio.wait_for(self.browser.close(), 10)
        finally:
            try:
                if self.playwright:
                    await self.playwright.stop()
            finally:
                await terminate(self.process)


async def doctor(options):
    """Check installed versions and actual WebMCP discovery without running a model."""
    import tempfile
    from importlib.metadata import version

    from .transport import Client

    executable = shutil.which(options["mcp_command"])
    if executable is None:
        raise FileNotFoundError(f"Install chrome-devtools-mcp@{MCP_VERSION} or use --mcp-command")
    options["mcp_command"] = str(Path(executable).resolve())
    versions = {"playwright": version("playwright")}
    commands = {
        "agent": [options["agent"], "--version"],
        "chrome": [chrome_path(options.get("chrome_path")), "--version"],
        "mcp": [options["mcp_command"], "--version"],
    }
    for name, command in commands.items():
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"{name} version check timed out") from None
        if process.returncode:
            raise RuntimeError(f"{name} version check failed: {stderr.decode()[-1000:]}")
        versions[name] = stdout.decode().strip()
    if versions["mcp"] != MCP_VERSION:
        raise ValueError(f"Install chrome-devtools-mcp@{MCP_VERSION}; found {versions['mcp']}")
    with tempfile.TemporaryDirectory(prefix="awa-doctor-") as root:
        browser = Browser(root, options)
        client = None
        try:
            await browser.start()
            client = await asyncio.wait_for(Client().start(browser.mcp_command()), 30)
            async with asyncio.timeout(30):
                listed = await client.request("tools/list", {})
                required = {"list_pages", "list_webmcp_tools", "execute_webmcp_tool"}
                if not required <= {t["name"] for t in listed["tools"]}:
                    raise RuntimeError("Chrome DevTools MCP lacks WebMCP tools")
                pages = await client.request("tools/call", {"name": "list_pages", "arguments": {}})
                if pages.get("isError"):
                    raise RuntimeError(json.dumps(pages))
            versions["webmcp_tools"] = sorted(browser.registered)
        finally:
            if client:
                await client.close()
            await browser.close()
    return versions
