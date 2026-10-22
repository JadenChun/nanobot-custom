# Crawl4AI worker-facing declarations

These local declarations describe only the four exports imported by
`nanobot/crawler/worker.py`, qualified against the installed Crawl4AI 0.9.4 wheel
(`46779c8a93b35c7c5f9be9642ddef5e7b515e36b094639bb8e1e418bf0266179`).

Config constructors deliberately cover the worker's keyword-only usage, not the
entire SDK or its positional API. Their named fields were checked against
`async_configs.py`; arbitrary configuration keywords are not admitted. Lists are
refined to the worker's string elements. SDK defaults of `None` are represented
where appropriate. Explicit `css_selector` remains string-only, matching the
SDK's parameter annotation: the worker omits an unset selector while the stored
attribute can consequently be `None`.

Crawler lifecycle/hook/session methods follow `async_webcrawler.py` and
`async_crawler_strategy.py`; `start()` returns the crawler itself. `arun()` keeps
an opaque `object` result because the installed implementation has container,
single-result and dynamically decorated paths. This is not a claim that its
result is validated or that the worker's existing dynamic SDK boundary is typed.

The stub does not change runtime imports, install dependencies or enable a
browser. Expand and requalify it when the worker adds SDK options or the frozen
distribution changes. Offline structural checks and mocked worker tests do not
substitute for the separately pending native/browser validation.
