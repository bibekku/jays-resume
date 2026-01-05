Below is the minimal, clean file layout with exact contents.

⸻

pytest.ini

[pytest]
testpaths = tests
addopts = -q
markers =
    ui: Playwright UI tests


⸻

tests/ui/conftest.py

import os
import pathlib
import pytest
from dataclasses import dataclass
from playwright.sync_api import (
    sync_playwright,
    Playwright,
    Browser,
    BrowserContext,
    Page,
    expect,
)

# ---------- Config ----------

@dataclass(frozen=True)
class UIConfig:
    base_url: str
    browser: str
    headless: bool
    slow_mo_ms: int
    viewport: tuple[int, int]

@pytest.fixture(scope="session")
def ui_config() -> UIConfig:
    return UIConfig(
        base_url=os.getenv("BASE_URL", "http://localhost:8080").rstrip("/"),
        browser=os.getenv("BROWSER", "chromium"),
        headless=os.getenv("HEADLESS", "1") != "0",
        slow_mo_ms=int(os.getenv("SLOWMO_MS", "0")),
        viewport=(1280, 720),
    )

# ---------- Playwright ----------

@pytest.fixture(scope="session")
def playwright_instance() -> Playwright:
    with sync_playwright() as p:
        yield p

@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright, ui_config: UIConfig) -> Browser:
    browser_type = getattr(playwright_instance, ui_config.browser)
    browser = browser_type.launch(
        headless=ui_config.headless,
        slow_mo=ui_config.slow_mo_ms,
    )
    yield browser
    browser.close()

# ---------- Auth State ----------

AUTH_DIR = pathlib.Path("test-results/auth")
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_STATE = AUTH_DIR / "storage_state.json"

@pytest.fixture(scope="session")
def ensure_auth_state(browser: Browser, ui_config: UIConfig):
    if os.getenv("STORAGE_STATE"):
        return

    if AUTH_STATE.exists() and os.getenv("REAUTH", "0") != "1":
        return

    ctx = browser.new_context(base_url=ui_config.base_url)
    page = ctx.new_page()

    page.goto("/login")
    page.get_by_label("Username").fill(os.environ["TEST_USERNAME"])
    page.get_by_label("Password").fill(os.environ["TEST_PASSWORD"])
    page.get_by_role("button", name="Sign in").click()

    expect(page).not_to_have_url(lambda u: "/login" in u)
    expect(page.get_by_role("navigation")).to_be_visible()

    ctx.storage_state(path=AUTH_STATE)
    ctx.close()

# ---------- Contexts ----------

@pytest.fixture
def context(
    browser: Browser,
    ui_config: UIConfig,
    ensure_auth_state,
):
    storage = os.getenv("STORAGE_STATE") or str(AUTH_STATE)
    ctx = browser.new_context(
        base_url=ui_config.base_url,
        storage_state=storage,
        viewport={"width": ui_config.viewport[0], "height": ui_config.viewport[1]},
        ignore_https_errors=True,
    )
    yield ctx
    ctx.close()

@pytest.fixture
def page(context: BrowserContext) -> Page:
    p = context.new_page()
    p.set_default_timeout(10_000)
    yield p
    p.close()

# ---------- Anonymous (Login Tests) ----------

@pytest.fixture
def anon_context(browser: Browser, ui_config: UIConfig):
    ctx = browser.new_context(
        base_url=ui_config.base_url,
        viewport={"width": ui_config.viewport[0], "height": ui_config.viewport[1]},
        ignore_https_errors=True,
    )
    yield ctx
    ctx.close()

@pytest.fixture
def anon_page(anon_context: BrowserContext) -> Page:
    p = anon_context.new_page()
    p.set_default_timeout(10_000)
    yield p
    p.close()


⸻

tests/ui/test_login.py

import os
import pytest
from playwright.sync_api import expect, Page

pytestmark = pytest.mark.ui

def test_redirects_to_login(anon_page: Page):
    anon_page.goto("/")
    expect(anon_page).to_have_url(lambda u: "/login" in u)

def test_login_success(anon_page: Page):
    anon_page.goto("/login")
    anon_page.get_by_label("Username").fill(os.environ["TEST_USERNAME"])
    anon_page.get_by_label("Password").fill(os.environ["TEST_PASSWORD"])
    anon_page.get_by_role("button", name="Sign in").click()

    expect(anon_page).not_to_have_url(lambda u: "/login" in u)
    expect(anon_page.get_by_role("navigation")).to_be_visible()

def test_login_failure(anon_page: Page):
    anon_page.goto("/login")
    anon_page.get_by_label("Username").fill("bad")
    anon_page.get_by_label("Password").fill("bad")
    anon_page.get_by_role("button", name="Sign in").click()

    expect(anon_page.get_by_text("Invalid")).to_be_visible()


⸻

tests/ui/test_smoke.py

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui

def test_home_page_loads(page: Page):
    page.goto("/")
    expect(page.get_by_role("navigation")).to_be_visible()


⸻

Environment Variables Required

BASE_URL=...
TEST_USERNAME=...
TEST_PASSWORD=...

Optional:

HEADLESS=0
SLOWMO_MS=150
REAUTH=1
STORAGE_STATE=/path/to/state.json


⸻

This is production-grade, codegen-friendly, and CI-safe.