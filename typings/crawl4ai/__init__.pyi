"""Worker-facing keyword API for the pinned Crawl4AI 0.9.4 distribution."""

from collections.abc import Callable
from enum import Enum
from typing import Protocol, Self

class CacheMode(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    BYPASS = "bypass"

class BrowserConfig:
    def __init__(
        self,
        *,
        browser_type: str = ...,
        channel: str = ...,
        headless: bool = ...,
        accept_downloads: bool = ...,
        ignore_https_errors: bool = ...,
        enable_stealth: bool = ...,
        memory_saving_mode: bool = ...,
        verbose: bool = ...,
        use_persistent_context: bool = ...,
        user_data_dir: str | None = ...,
    ) -> None: ...

class CrawlerRunConfig:
    css_selector: str | None
    def __init__(
        self,
        *,
        cache_mode: CacheMode = ...,
        session_id: str | None = ...,
        js_only: bool = ...,
        js_code: str | list[str] | None = ...,
        css_selector: str = ...,
        page_timeout: int = ...,
        delay_before_return_html: float = ...,
        wait_until: str = ...,
        wait_for_images: bool = ...,
        screenshot: bool = ...,
        force_viewport_screenshot: bool = ...,
        image_score_threshold: int = ...,
        image_description_min_word_threshold: int = ...,
        excluded_tags: list[str] | None = ...,
        keep_attrs: list[str] | None = ...,
        keep_data_attributes: bool = ...,
        remove_overlay_elements: bool = ...,
        verbose: bool = ...,
    ) -> None: ...

class _CrawlerStrategy(Protocol):
    browser_manager: object
    def set_hook(self, hook_type: str, hook: Callable[..., object]) -> None: ...
    async def kill_session(self, session_id: str) -> None: ...

class AsyncWebCrawler:
    crawler_strategy: _CrawlerStrategy
    def __init__(self, *, config: BrowserConfig | None = ..., **kwargs: object) -> None: ...
    async def start(self) -> Self: ...
    async def close(self) -> None: ...
    async def arun(
        self, url: str, config: CrawlerRunConfig | None = ..., **kwargs: object,
    ) -> object: ...
