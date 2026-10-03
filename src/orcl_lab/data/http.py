"""HTTP helpers: a retrying session and SEC-compliant headers (user agent from the environment)."""
from __future__ import annotations

import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_SEC_UA = "ORCL-Quant-Lab educational research project (set SEC_USER_AGENT env var)"


def sec_headers() -> dict[str, str]:
    """SEC asks automated clients to identify themselves; set SEC_USER_AGENT='Name email@domain'."""
    return {"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_UA), "Accept-Encoding": "gzip, deflate"}


def make_session(headers: dict[str, str] | None = None, retries: int = 4, backoff: float = 1.0) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    if headers:
        s.headers.update(headers)
    return s


class PoliteClient:
    """Session wrapper that spaces requests out (SEC fair-access: <= 10 requests/second)."""

    def __init__(self, headers: dict[str, str] | None = None, pause: float = 0.25):
        self.session = make_session(headers)
        self.pause = pause
        self._last = 0.0

    def get(self, url: str, **kw) -> requests.Response:
        wait = self.pause - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=kw.pop("timeout", 60), **kw)
        self._last = time.monotonic()
        resp.raise_for_status()
        return resp
