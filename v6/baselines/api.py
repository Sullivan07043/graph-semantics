"""Small, judge-independent OpenAI client for LLM-based baselines.

The client intentionally uses only the Python standard library.  Each successful
request is cached as one JSON file, so interrupted baseline runs can resume
without paying for the same prompt twice and can reconstruct their API spend.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Optional
import urllib.error
import urllib.request


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60


class BaselineAPIError(RuntimeError):
    """The baseline API request or response was invalid."""


class BaselineAuthenticationError(BaselineAPIError):
    """The API rejected the supplied key; retrying cannot fix this."""


class BaselineBudgetExceeded(BaselineAPIError):
    """A new request would exceed the configured run budget."""


def available(api_key: Optional[str] = None) -> bool:
    """Return whether an API key is available without sending a request."""

    return bool(api_key or os.environ.get("OPENAI_API_KEY"))


def make_cache_key(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Optional[Mapping[str, Any]],
    prompt_version: str,
) -> str:
    """Return a stable key covering every input that can change a completion."""

    material = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "schema": schema,
        "prompt_version": prompt_version,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_count(usage: Mapping[str, Any], primary: str, alternate: str) -> int:
    value = usage.get(primary, usage.get(alternate))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise BaselineAPIError(f"API response has invalid usage.{primary}")
    return int(value)


def _usage_cost(
    usage: Mapping[str, Any], input_rate: float, output_rate: float
) -> float:
    input_tokens = _token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens = _token_count(usage, "completion_tokens", "output_tokens")
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0


class BaselineAPIClient:
    """Cached Chat Completions client used only by generation baselines.

    ``transport`` is injectable for tests.  It has the same calling convention
    as ``urllib.request.urlopen``: ``transport(request, timeout=seconds)`` and
    returns an object with ``read()`` (and optionally ``status``/``close``).
    """

    def __init__(
        self,
        cache_dir: os.PathLike[str] | str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        budget_usd: Optional[float] = None,
        timeout: float = 120.0,
        max_attempts: int = 4,
        retry_base_seconds: float = 1.0,
        transport: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        input_usd_per_million: float = INPUT_USD_PER_MILLION,
        output_usd_per_million: float = OUTPUT_USD_PER_MILLION,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget_usd must be non-negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if timeout <= 0 or retry_base_seconds < 0:
            raise ValueError("timeout must be positive and retry delay non-negative")
        if input_usd_per_million < 0 or output_usd_per_million < 0:
            raise ValueError("token prices must be non-negative")

        self.cache_dir = Path(cache_dir)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("BASELINE_API_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.budget_usd = budget_usd
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.retry_base_seconds = float(retry_base_seconds)
        self.transport = transport or urllib.request.urlopen
        self.sleep = sleep
        self.input_usd_per_million = float(input_usd_per_million)
        self.output_usd_per_million = float(output_usd_per_million)
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._scan_cache()

    @property
    def spent_usd(self) -> float:
        """Actual cost reconstructed from successful cached responses."""

        with self._lock:
            return self._spent_usd

    @property
    def remaining_usd(self) -> Optional[float]:
        with self._lock:
            if self.budget_usd is None:
                return None
            return max(0.0, self.budget_usd - self._spent_usd - self._reserved_usd)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Mapping[str, Any]],
        prompt_version: str,
        max_output_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Return a JSON completion plus provenance, using the cache on a hit.

        The returned dictionary contains ``data`` (parsed JSON), ``raw_content``,
        requested/returned model names, usage, timestamp, cost, cache key, and a
        transient ``cached`` flag.  The flag is not written to disk.
        """

        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise TypeError("system_prompt and user_prompt must be strings")
        if not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError("prompt_version must be a non-empty string")
        if isinstance(max_output_tokens, bool) or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        key = make_cache_key(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            prompt_version=prompt_version,
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return self._result(cached, cached=True)

        if not self.api_key:
            raise BaselineAuthenticationError(
                "OPENAI_API_KEY is not set for the baseline API client"
            )

        if schema is None:
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "baseline_result",
                    "strict": True,
                    "schema": copy.deepcopy(dict(schema)),
                },
            }
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": int(max_output_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reservation = self._reserve_budget(len(body), max_output_tokens)
        reservation_released = False
        try:
            response = self._request_with_retries(body)

            try:
                returned_model = response["model"]
                usage = response["usage"]
                raw_content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise BaselineAPIError("API response is missing completion metadata") from exc
            if not isinstance(returned_model, str) or not isinstance(usage, Mapping):
                raise BaselineAPIError("API response has invalid model or usage metadata")
            if not isinstance(raw_content, str):
                raise BaselineAPIError("API response content is not a string")
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                raise BaselineAPIError("API response content is not valid JSON") from exc

            cost = _usage_cost(
                usage, self.input_usd_per_million, self.output_usd_per_million
            )
            record = {
                "cache_key": key,
                "prompt_version": prompt_version,
                "request": {
                    "endpoint": f"{self.base_url}/chat/completions",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "schema": copy.deepcopy(dict(schema)) if schema is not None else None,
                    "temperature": 0,
                    "max_output_tokens": int(max_output_tokens),
                },
                "requested_model": self.model,
                "returned_model": returned_model,
                "usage": copy.deepcopy(dict(usage)),
                "pricing_usd_per_million": {
                    "input": self.input_usd_per_million,
                    "output": self.output_usd_per_million,
                },
                "cost_usd": cost,
                "raw_content": raw_content,
                "data": data,
                "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
            self._write_record(record)
            with self._lock:
                self._cache[key] = record
                self._spent_usd += cost
                self._reserved_usd -= reservation
                reservation_released = True
                spent = self._spent_usd
            if self.budget_usd is not None and spent > self.budget_usd + 1e-12:
                # The response is cached because its cost has already been incurred.
                raise BaselineBudgetExceeded(
                    f"API response raised spend to ${spent:.6f}, above the "
                    f"${self.budget_usd:.6f} baseline budget"
                )
            return self._result(record, cached=False)
        finally:
            if not reservation_released and reservation:
                with self._lock:
                    self._reserved_usd -= reservation

    def _scan_cache(self) -> None:
        if not self.cache_dir.is_dir():
            return
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                key = record["cache_key"]
                usage = record["usage"]
                pricing = record.get("pricing_usd_per_million", {})
                input_rate = float(pricing.get("input", INPUT_USD_PER_MILLION))
                output_rate = float(pricing.get("output", OUTPUT_USD_PER_MILLION))
                if not isinstance(key, str) or path.stem != key:
                    continue
                cost = _usage_cost(usage, input_rate, output_rate)
                record["cost_usd"] = cost
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
                    BaselineAPIError):
                continue
            if key not in self._cache:
                self._cache[key] = record
                self._spent_usd += cost

    def _reserve_budget(
        self, serialized_request_bytes: int, max_output_tokens: int
    ) -> float:
        if self.budget_usd is None:
            return 0.0
        # A byte is a conservative upper bound for one input token; the fixed
        # allowance covers ChatML framing not present in the serialized request.
        input_token_bound = serialized_request_bytes + 256
        reserved = (
            input_token_bound * self.input_usd_per_million
            + max_output_tokens * self.output_usd_per_million
        ) / 1_000_000.0
        with self._lock:
            remaining = self.budget_usd - self._spent_usd - self._reserved_usd
            if reserved > remaining + 1e-12:
                raise BaselineBudgetExceeded(
                    f"request reserves up to ${reserved:.6f}, but only "
                    f"${max(0.0, remaining):.6f} remains in the baseline budget"
                )
            self._reserved_usd += reserved
        return reserved

    def _request_with_retries(self, body: bytes) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_attempts):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                response = self.transport(request, timeout=self.timeout)
                status = int(getattr(response, "status", 200))
                raw = response.read()
                close = getattr(response, "close", None)
                if close is not None:
                    close()
                if status >= 400:
                    raise _StatusError(status, raw, getattr(response, "headers", {}))
                parsed = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                if not isinstance(parsed, dict):
                    raise BaselineAPIError("API response body is not a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                status = exc.code
                headers = exc.headers or {}
                retry = status == 429 or 500 <= status < 600
                message = raw.decode("utf-8", errors="replace")
            except _StatusError as exc:
                status, headers = exc.status, exc.headers
                retry = status == 429 or 500 <= status < 600
                message = exc.body.decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                status, headers, retry = None, {}, True
                message = str(exc)

            if status == 401:
                raise BaselineAuthenticationError(
                    "OpenAI authentication failed (HTTP 401); the response body is "
                    "suppressed because it may echo credential fragments"
                )
            if not retry:
                raise BaselineAPIError(
                    f"OpenAI API failed (HTTP {status}): {message[:500]}"
                )
            if attempt + 1 >= self.max_attempts:
                label = f"HTTP {status}" if status is not None else "network error"
                raise BaselineAPIError(
                    f"OpenAI API failed after {self.max_attempts} attempts "
                    f"({label}): {message[:500]}"
                )
            self.sleep(self._retry_delay(attempt, headers))
        raise AssertionError("unreachable")

    def _retry_delay(self, attempt: int, headers: Any) -> float:
        try:
            retry_after = float(headers.get("Retry-After"))
            if retry_after >= 0:
                return min(retry_after, 60.0)
        except (AttributeError, TypeError, ValueError):
            pass
        return min(self.retry_base_seconds * (2 ** attempt), 60.0)

    def _write_record(self, record: Mapping[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"{record['cache_key']}.json"
        fd, temporary = tempfile.mkstemp(
            dir=self.cache_dir, prefix=f".{record['cache_key']}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _result(record: Mapping[str, Any], *, cached: bool) -> dict[str, Any]:
        result = copy.deepcopy(dict(record))
        result["cached"] = cached
        return result


class _StatusError(Exception):
    def __init__(self, status: int, body: bytes, headers: Any) -> None:
        self.status = status
        self.body = body
        self.headers = headers


__all__ = [
    "BaselineAPIClient",
    "BaselineAPIError",
    "BaselineAuthenticationError",
    "BaselineBudgetExceeded",
    "DEFAULT_MODEL",
    "available",
    "make_cache_key",
]
