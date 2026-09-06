"""Optional real-browser checks: AWA_TEST_BROWSER=/path/to/chrome uv run pytest …"""

import copy
import json
import os
from dataclasses import asdict

import pytest

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.vis import results
from assembly_world_agent.vis.replay import read_episode


@pytest.mark.skipif(not os.environ.get("AWA_TEST_BROWSER"), reason="Requires a browser executable")
def test_offline_controls(row, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    source = adapt_sample("ikea-manual", row, revision="fixture")
    prepared = prepare_sample(source)
    ep = read_episode(export_episode(prepared, tmp_path / "initial.zip").path)
    frame = copy.deepcopy(ep["states"][0])
    frame["camera"]["position"][0] += 1
    ep["states"][1] = frame
    ep["calls"] = [
        dict(
            index=0,
            name="get_scene",
            arguments={},
            status="completed",
            before_index=0,
            state_index=0,
            result={"text": "</script><script>window.injected=true</script>"},
        ),
        dict(
            index=1,
            name="move_camera",
            arguments={"position": [1, 2, 3]},
            status="completed",
            before_index=0,
            state_index=1,
        ),
        dict(
            index=2,
            name="bad_call",
            arguments={},
            status="failed",
            before_index=1,
            state_index=1,
            error="Expected fixture failure",
        ),
    ]
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "identity": {
                        "dataset": "ikea-manual",
                        "revision": "fixture",
                        "preparation": asdict(prepared.config),
                    },
                    "samples": {
                        source.sample_id: {"episode_id": ep["manifest"]["id"]},
                        "Missing": {},
                    },
                }
            }
        )
    )
    monkeypatch.setattr(results, "load_samples", lambda *a, **kw: iter([source]))
    monkeypatch.setattr(
        results,
        "select_episode",
        lambda p, expected_id=None: (
            (None, None, [])
            if p.name == "Missing"
            else (ep, {"path": "final.episode.zip", "sha256": "fixture"}, [])
        ),
    )
    monkeypatch.setattr(
        results,
        "manual_pages",
        lambda *a: [
            {
                "data": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'><rect width='40' height='40' fill='red'/></svg>"
            },
            {
                "data": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'><rect width='40' height='40' fill='blue'/></svg>"
            },
        ],
    )
    output = tmp_path / "result.html"
    results.export_results(run, output)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ["AWA_TEST_BROWSER"], headless=True)
        context = browser.new_context(offline=True, viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        errors, requests = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url) if r.url.startswith("http") else None)
        page.goto(output.as_uri())
        page.wait_for_function("window.resultViewer?.rows[0]?.replay")
        assert page.locator(".sample").count() == 2
        page.locator("#last").click()
        assert page.locator(".call").first.inner_text().startswith("bad_call")
        page.locator("#mode").select_option("changes")
        assert page.locator(".call").first.inner_text().startswith("move_camera")
        page.locator("#first").click()
        assert page.locator(".call").first.inner_text().startswith("Initial state")
        page.locator("#loop").click()
        page.wait_for_timeout(1150)
        assert page.locator(".call").first.inner_text().startswith("move_camera")
        page.locator("#pause").click()
        page.locator("#mode").select_option("all")
        cell = page.locator(".sample").first.locator(".cell").first
        cell.get_by_role("button", name="Next", exact=True).click()
        assert "2 / 2" in cell.inner_text()
        cell.locator("img").click()
        assert page.locator("#zoom").is_visible()
        page.locator("#close-zoom").click()
        view = page.locator(".view").nth(1).bounding_box()
        page.mouse.move(view["x"] + 80, view["y"] + 80)
        page.mouse.down()
        page.mouse.move(view["x"] + 130, view["y"] + 110)
        page.mouse.up()
        assert not page.evaluate("resultViewer.rows[0].follow")
        page.locator("#follow").uncheck()
        page.locator("#follow").check()
        assert page.evaluate("resultViewer.rows[0].follow")
        page.evaluate("""() => {
            const spacer = document.createElement('div');
            spacer.style.height = '2000px'; document.body.append(spacer);
            window.scrollTo(0, document.body.scrollHeight);
        }""")
        page.wait_for_function("resultViewer.rows[0].data === null")
        page.locator("#last").click()
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_function("Boolean(resultViewer.rows[0].replay)")
        assert page.locator(".call").first.inner_text().startswith("bad_call")
        assert "2 / 2" in cell.inner_text()
        assert not page.evaluate("Boolean(window.injected)")
        assert not errors and not requests
        assert not page.evaluate("resultViewer.renderer.getContext().isContextLost()")
        page.screenshot(path=str(tmp_path / "result.png"), full_page=True)
        browser.close()
