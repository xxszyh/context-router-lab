"""Retrying the transport, not the answer.

A judge run is ``2 * repeats * pairs`` calls and takes tens of minutes; the sea-ice
conversation at three repeats is 264 calls and ran for 51 minutes. A single read timeout at
call 200 of those used to discard the whole run *and* the whole bill, because the CLI writes
its artefact once, at the end. That is not hypothetical -- it happened on 2026-10-01 and is
why this module exists.

**Only transport failures are retried.** ``httpx.HTTPStatusError`` is deliberately excluded,
and that exclusion is load-bearing: a 400 is the server's answer rather than a hiccup, and one
of this project's results is measured in exactly those. `full_history` failing to run against a
documented model's context ceiling is a count of HTTP 400s; retrying them would turn a finding
into a timeout and quietly delete the finding.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

#: Failures that mean "the request did not complete", as opposed to "the server answered".
TRANSIENT = (httpx.TimeoutException, httpx.TransportError)


def post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    attempts: int = 4,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """POST and return a response whose status has already been checked.

    Returning only after ``raise_for_status`` keeps each caller's error handling identical to
    the single-attempt version: a status error propagates on the first attempt, and only a
    transport error is given another try. ``sleep`` is injectable so a test does not wait out
    the backoff.
    """

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last: Exception = RuntimeError("no attempt was made")
    for attempt in range(attempts):
        try:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response
        except TRANSIENT as error:
            last = error
            if attempt + 1 < attempts:
                sleep(backoff * (2**attempt))
    raise last
