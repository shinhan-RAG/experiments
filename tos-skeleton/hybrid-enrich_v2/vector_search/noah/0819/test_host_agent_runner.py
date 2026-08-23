import json
import unittest

import host_agent_runner


class HostAgentRunnerTest(unittest.TestCase):
    def test_claude_usage_envelope_is_normalized(self):
        envelope = {
            "total_cost_usd": 0.0123,
            "usage": {
                "input_tokens": 120,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
                "output_tokens": 50,
                "output_tokens_details": {"thinking_tokens": 7},
            },
        }
        self.assertEqual(
            {"input_tokens": 120, "cached_input_tokens": 30,
             "cache_write_input_tokens": 40, "output_tokens": 50,
             "reasoning_output_tokens": 7, "cost_usd": 0.0123},
            host_agent_runner.model_usage(json.dumps(envelope), "claude"),
        )

    def test_invalid_claude_envelope_has_no_usage(self):
        self.assertEqual({}, host_agent_runner.model_usage("not-json", "claude"))


if __name__ == "__main__":
    unittest.main()
