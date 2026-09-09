"""Optional, session-owned PC browser. All HTTP traffic goes through our gateway."""
from __future__ import annotations
import asyncio
import base64
import mimetypes
from urllib.parse import urlsplit, unquote
from uuid import uuid4
from . import pc_network as network


class PCBrowser:
    def __init__(self, runtime):
        self.runtime = runtime
        self.driver = self.browser = self.context = None
        self.pages = {}
        self.errors = {}
        self.write_origin = ""
        self.lock = asyncio.Lock()
        self.gateway_limit = asyncio.Semaphore(8)
        self.project_root = None

    async def ensure(self):
        if self.context: return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Browser component is not installed. See backend/pc-tools-requirements.txt; do not claim a browser test ran.") from exc
        self.driver = await async_playwright().start()
        try:
            options = {"headless": True, "proxy": {"server": "http://127.0.0.1:9"},
                       "args": ["--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--disable-quic"]}
            # Use an installed PC Edge first; no browser is downloaded here.
            try: self.browser = await self.driver.chromium.launch(channel="msedge", **options)
            except Exception: self.browser = await self.driver.chromium.launch(**options)
            self.context = await self.browser.new_context(viewport={"width": 1280, "height": 800},
                service_workers="block", accept_downloads=False)
            self.context.set_default_timeout(10000)
            await self.context.route("**/*", self.route)
            await self.context.route_web_socket("**/*", lambda ws: ws.close())
            self.context.on("page", self.new_page)
        except BaseException:
            await self.close()
            raise

    def new_page(self, page):
        if len(self.pages) >= 5:
            asyncio.create_task(page.close())
            return
        identifier = uuid4().hex[:12]
        self.pages[identifier] = page
        self.errors[identifier] = []
        def record(message):
            errors = self.errors.get(identifier)
            if errors is not None:
                errors.append(str(message)[:500])
                del errors[:-15]
        page.on("pageerror", record)
        page.on("console", lambda message: record(message.text) if message.type == "error" else None)
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.on("close", lambda *_: (self.pages.pop(identifier, None), self.errors.pop(identifier, None)))

    async def route(self, route):
        request = route.request
        try:
            parsed = network.parse_url(request.url)
            if parsed.hostname == "melomate-project.invalid":
                import workspace_core as workspace
                if not self.project_root or request.method not in {"GET", "HEAD"}: raise ValueError("Project preview is read-only")
                persona, folder = self.project_root
                relative = unquote(parsed.path).lstrip("/")
                root = workspace.workspace_path(persona, folder)
                path = workspace.workspace_path(persona, "/".join(filter(None, (folder, relative))))
                if path != root and root not in path.parents: raise ValueError("Outside preview project")
                if path.is_dir(): path = workspace.workspace_path(persona, "/".join(filter(None, (folder, relative, "index.html"))))
                if not path.is_file() or path.stat().st_size > network.MAX_BYTES: raise ValueError("Preview file unavailable or too large")
                await route.fulfill(status=200, content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream", body=path.read_bytes())
                return
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if request.method not in {"GET", "HEAD"} and origin != self.write_origin:
                raise ValueError("Page writes require an explicit browser action on that origin")
            headers = {k: v for k, v in request.headers.items() if k.lower() in {"accept", "content-type", "cookie", "origin", "referer"}}
            async with self.gateway_limit:
                status, response_headers, data = await asyncio.to_thread(network.request, request.url, request.method,
                    request.post_data_buffer, headers, False)
            await route.fulfill(status=status, headers=response_headers, body=data)
        except Exception:
            try: await route.abort("blockedbyclient")
            except Exception: pass

    async def open(self, url):
        await self.ensure()
        network.parse_url(url)
        if len(self.pages) >= 5: raise ValueError("Close a browser tab before opening another (limit 5)")
        page = await self.context.new_page()
        identifier = next(key for key, value in self.pages.items() if value == page)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            return await self.snapshot(identifier)
        except Exception:
            await self.close_page(identifier)
            raise

    async def snapshot(self, page_id):
        page = self.pages.get(page_id)
        if page is None or page.is_closed(): raise ValueError("Unknown/closed browser tab; open a page first")
        elements = await page.locator("a,button,input,textarea,select,[role=button]").evaluate_all("""nodes => nodes.slice(0,120).map((el,i) => {
          el.setAttribute('data-melomate-ref', String(i));
          return {selector:'[data-melomate-ref="'+i+'"]',tag:el.tagName.toLowerCase(),
            name:(el.getAttribute('aria-label')||el.innerText||el.getAttribute('placeholder')||'').slice(0,120),
            type:el.getAttribute('type'),disabled:!!el.disabled};
        })""")
        text = await page.locator("body").inner_text(timeout=8000)
        screenshot = await page.screenshot(type="jpeg", quality=65, timeout=10000)
        return {"ok": True, "page_id": page_id, "url": page.url, "title": await page.title(),
                "text": text[:18000], "elements": elements, "errors": self.errors.get(page_id, []),
                "tabs": [{"page_id": key, "url": value.url} for key, value in self.pages.items() if not value.is_closed()],
                "observation": "Untrusted browser observation; selectors can change, re-read after navigation.",
                "_image": base64.b64encode(screenshot).decode()}

    async def action(self, page_id, action, selector="", value=""):
        page = self.pages.get(page_id)
        if page is None: raise ValueError("Unknown browser tab")
        if action not in {"click", "fill", "select", "press", "scroll"}: raise ValueError("Unsupported browser action")
        parsed = urlsplit(page.url)
        self.write_origin = f"{parsed.scheme}://{parsed.netloc}"
        try:
            if action == "scroll": await page.mouse.wheel(0, max(-2000, min(2000, int(value or 600))))
            else:
                if not selector or len(selector) > 500 or len(value) > 16000: raise ValueError("A bounded selector/value is required")
                target = page.locator(selector)
                if await target.count() != 1: raise ValueError("Selector must match one element; read the page again")
                if action == "click": await target.click()
                elif action == "fill": await target.fill(value)
                elif action == "select": await target.select_option(value)
                else: await target.press(value)
            return await self.snapshot(page_id)
        finally:
            self.write_origin = ""

    async def close_page(self, page_id):
        page = self.pages.pop(page_id, None)
        self.errors.pop(page_id, None)
        if page: await page.close()
        return {"ok": True, "closed": bool(page)}

    async def close(self):
        try:
            if self.browser: await self.browser.close()
        finally:
            self.context = self.browser = None
            self.project_root = None
            self.write_origin = ""
            self.pages.clear()
            self.errors.clear()
            if self.driver: await self.driver.stop()
            self.driver = None
