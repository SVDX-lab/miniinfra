"""Deterministic conversations that exercise Qwen3 template boundaries."""

BOUNDARY_CASES = {
    "multi_turn": {
        "source_line": -201,
        "conversations": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
    },
    "system_prompt": {
        "source_line": -202,
        "conversations": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    },
    "tool_call_and_response": {
        "source_line": -203,
        "conversations": [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": {"city": "杭州"}}}
                ],
            },
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "晴天。"},
        ],
    },
    "empty_assistant": {
        "source_line": -204,
        "conversations": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": ""},
        ],
    },
}
