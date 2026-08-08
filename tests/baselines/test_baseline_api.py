import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from v6.baselines.api import (
    BaselineAPIClient,
    BaselineAuthenticationError,
    BaselineBudgetExceeded,
    make_cache_key,
)


SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.closed = False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        self.closed = True


def completion(*, prompt_tokens=100, completion_tokens=20):
    return {
        "model": "gpt-4o-mini-2024-07-18",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "choices": [{"message": {"content": '{"label":"openness"}'}}],
    }


class BaselineAPIClientTests(unittest.TestCase):
    def test_payload_metadata_atomic_cache_and_cache_hit(self):
        requests = []

        def transport(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(completion())

        with tempfile.TemporaryDirectory() as directory:
            client = BaselineAPIClient(
                directory, api_key="test-key", transport=transport, sleep=lambda _: None
            )
            first = client.complete_json(
                system_prompt="Name the feature.",
                user_prompt="top examples",
                schema=SCHEMA,
                prompt_version="autointerp-v1",
            )

            self.assertEqual(len(requests), 1)
            request, timeout = requests[0]
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "gpt-4o-mini")
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(
                payload["response_format"]["json_schema"]["schema"], SCHEMA
            )
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertEqual(timeout, 120.0)
            self.assertEqual(first["requested_model"], "gpt-4o-mini")
            self.assertEqual(first["returned_model"], "gpt-4o-mini-2024-07-18")
            self.assertEqual(first["data"], {"label": "openness"})
            self.assertEqual(first["raw_content"], '{"label":"openness"}')
            self.assertEqual(first["request"]["system_prompt"], "Name the feature.")
            self.assertEqual(first["request"]["user_prompt"], "top examples")
            self.assertEqual(first["request"]["temperature"], 0)
            self.assertEqual(first["usage"]["prompt_tokens"], 100)
            self.assertRegex(first["timestamp"], r"^\d{4}-\d\d-\d\dT.*Z$")
            self.assertFalse(first["cached"])
            self.assertAlmostEqual(first["cost_usd"], 0.000027)

            cache_files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            on_disk = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertNotIn("cached", on_disk)
            self.assertEqual(on_disk["cache_key"], cache_files[0].stem)

            second = client.complete_json(
                system_prompt="Name the feature.",
                user_prompt="top examples",
                schema=SCHEMA,
                prompt_version="autointerp-v1",
            )
            self.assertEqual(len(requests), 1)
            self.assertTrue(second["cached"])

    def test_cache_key_covers_all_prompt_inputs(self):
        base = {
            "model": "gpt-4o-mini",
            "system_prompt": "system",
            "user_prompt": "user",
            "schema": SCHEMA,
            "prompt_version": "v1",
        }
        keys = {make_cache_key(**base)}
        changes = {
            "model": "another-model",
            "system_prompt": "another system",
            "user_prompt": "another user",
            "schema": {"type": "array"},
            "prompt_version": "v2",
        }
        for field, value in changes.items():
            candidate = dict(base)
            candidate[field] = value
            keys.add(make_cache_key(**candidate))
        self.assertEqual(len(keys), 6)

        reordered = {
            "additionalProperties": False,
            "required": ["label"],
            "properties": {"label": {"type": "string"}},
            "type": "object",
        }
        candidate = dict(base)
        candidate["schema"] = reordered
        self.assertEqual(make_cache_key(**base), make_cache_key(**candidate))

    def test_retries_rate_limit_and_network_errors(self):
        events = [
            urllib.error.HTTPError(
                "https://api.openai.com/v1/chat/completions",
                429,
                "rate limited",
                {"Retry-After": "0"},
                io.BytesIO(b'{"error":"rate limited"}'),
            ),
            urllib.error.URLError("temporary network failure"),
            FakeResponse(completion()),
        ]
        calls = []
        delays = []

        def transport(request, timeout):
            calls.append(request)
            event = events.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

        with tempfile.TemporaryDirectory() as directory:
            client = BaselineAPIClient(
                directory,
                api_key="test-key",
                transport=transport,
                sleep=delays.append,
                retry_base_seconds=0.25,
            )
            result = client.complete_json(
                system_prompt="system",
                user_prompt="user",
                schema=SCHEMA,
                prompt_version="v1",
            )
        self.assertEqual(result["data"]["label"], "openness")
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [0.0, 0.5])

    def test_does_not_retry_401(self):
        calls = []
        delays = []

        def transport(request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized",
                {},
                io.BytesIO(b'{"error":"invalid key"}'),
            )

        with tempfile.TemporaryDirectory() as directory:
            client = BaselineAPIClient(
                directory,
                api_key="invalid",
                transport=transport,
                sleep=delays.append,
            )
            with self.assertRaises(BaselineAuthenticationError):
                client.complete_json(
                    system_prompt="system",
                    user_prompt="user",
                    schema=SCHEMA,
                    prompt_version="v1",
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(delays, [])

    def test_resume_scans_actual_usage_and_enforces_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BaselineAPIClient(
                directory,
                api_key="test-key",
                transport=lambda request, timeout: FakeResponse(
                    completion(prompt_tokens=1_000_000, completion_tokens=1_000_000)
                ),
                sleep=lambda _: None,
            )
            writer.complete_json(
                system_prompt="system",
                user_prompt="already paid",
                schema=SCHEMA,
                prompt_version="v1",
            )
            self.assertAlmostEqual(writer.spent_usd, 0.75)

            calls = []
            resumed = BaselineAPIClient(
                directory,
                api_key="test-key",
                budget_usd=0.75,
                transport=lambda request, timeout: calls.append(request),
                sleep=lambda _: None,
            )
            self.assertAlmostEqual(resumed.spent_usd, 0.75)
            self.assertEqual(resumed.remaining_usd, 0.0)

            cached = resumed.complete_json(
                system_prompt="system",
                user_prompt="already paid",
                schema=SCHEMA,
                prompt_version="v1",
            )
            self.assertTrue(cached["cached"])
            with self.assertRaises(BaselineBudgetExceeded):
                resumed.complete_json(
                    system_prompt="system",
                    user_prompt="new request",
                    schema=SCHEMA,
                    prompt_version="v1",
                )
            self.assertEqual(calls, [])

    def test_module_has_no_judge_dependency(self):
        module_path = (
            Path(__file__).parents[2] / "v6" / "baselines" / "api.py"
        )
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("import judge", source)
        self.assertNotIn("from .judge", source)
        self.assertNotIn("from v6.judge", source)


if __name__ == "__main__":
    unittest.main()
