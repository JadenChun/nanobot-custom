"""Web tools: web_search and web_fetch."""

import html
import json
import os
import re
import asyncio
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from playwright.async_api import async_playwright
import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.max_results = max_results
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        try:
            n = min(max(count or self.max_results, 1), 10)
            
            # Using run_in_executor to not block the async event loop with sync DDGS
            def sync_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=n))
                    
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, sync_search)
                
            if not results:
                return f"No results for: {query}"
            
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('href', '')}")
                if desc := item.get("body"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent=USER_AGENT,
                    java_script_enabled=True
                )
                
                # Setup timeout and wait for load
                try:
                    res = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception as goto_err:
                    await browser.close()
                    raise Exception(f"Failed to load page: {goto_err}")
                
                if not res:
                    await browser.close()
                    raise Exception("No response received from the page")
                
                ctype = res.headers.get("content-type", "")
                
                # Let pages load dynamic content
                await page.wait_for_timeout(2000) 
                
                # JSON
                if "application/json" in ctype:
                    body_text = await page.evaluate("() => document.body.innerText")
                    text, extractor = body_text, "json"
                # HTML
                elif "text/html" in ctype or (await page.content())[:256].lower().startswith(("<!doctype", "<html")):
                    html_content = await page.content()
                    doc = Document(html_content)
                    
                    if extractMode == "markdown":
                        import markdownify
                        content = markdownify.markdownify(doc.summary(), heading_style="ATX")
                    else:
                        content = _strip_tags(doc.summary())
                        
                    title = doc.title() or await page.title()
                    text = f"# {title}\n\n{content}" if title else content
                    extractor = "readability+playwright"
                else:
                    text, extractor = await page.content(), "raw+playwright"
                
                final_url = page.url
                status = res.status
                await browser.close()

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(final_url), "status": status,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
