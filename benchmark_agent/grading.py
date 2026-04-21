from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return value


def _args_equal(left: dict[str, Any], right: dict[str, Any], mode: str) -> bool:
    if mode == "exact":
        return left == right
    if mode == "canonical_json":
        return json.dumps(_canonicalize(left), sort_keys=True) == json.dumps(
            _canonicalize(right), sort_keys=True
        )
    if mode == "schema_only":
        return set(left.keys()) == set(right.keys())
    raise ValueError(f"Unknown args_match mode: {mode}")


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    if not small:
        return True
    j = 0
    for x in big:
        if j < len(small) and x == small[j]:
            j += 1
    return j == len(small)


def _names_match(
    actual_names: list[str],
    expected_names: list[str],
    *,
    match_mode: str,
    allow_extra: bool,
) -> bool:
    if match_mode == "strict_order":
        if allow_extra:
            return _is_subsequence(expected_names, actual_names)
        return actual_names == expected_names
    if match_mode == "multiset":
        ca, ce = Counter(actual_names), Counter(expected_names)
        if allow_extra:
            return ce <= ca
        return ca == ce
    if match_mode == "subset":
        if not _is_subsequence(expected_names, actual_names):
            return False
        if allow_extra:
            return True
        return Counter(actual_names) == Counter(expected_names)
    raise ValueError(f"Unknown match_mode: {match_mode}")


def grade_tool_trace(
    actual: list[dict[str, Any]],
    *,
    expected_tool_names: list[str],
    match_mode: str = "strict_order",
    allow_extra: bool = False,
    expected_args: list[dict[str, Any]] | None = None,
    args_match: str = "canonical_json",
    forbidden_tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    SSOT `actual` entries: {"name": str, "arguments": dict}.
    Returns flags suitable for logging / assertions.
    """
    actual_names = [x.get("name", "") for x in actual]
    forbidden = set(forbidden_tool_names or ())
    forbidden_hits = [n for n in actual_names if n in forbidden]

    names_ok = _names_match(actual_names, expected_tool_names, match_mode=match_mode, allow_extra=allow_extra)
    args_ok = True
    args_detail: list[dict[str, Any]] = []

    if expected_args is not None and expected_args:
        if len(actual) < len(expected_args):
            args_ok = False
            args_detail.append({"reason": "too_few_calls", "expected": len(expected_args), "actual": len(actual)})
        for i, exp_arg in enumerate(expected_args):
            if i >= len(actual):
                break
            an = actual[i].get("name")
            if i < len(expected_tool_names) and an != expected_tool_names[i]:
                args_ok = False
                args_detail.append({"index": i, "reason": "name_mismatch_for_args_slot"})
                continue
            aargs = actual[i].get("arguments") or {}
            if not isinstance(aargs, dict):
                aargs = {}
            ok = _args_equal(aargs, exp_arg, args_match)
            args_detail.append({"index": i, "ok": ok})
            if not ok:
                args_ok = False

    extra_calls = 0
    if not allow_extra and len(actual_names) != len(expected_tool_names):
        extra_calls = abs(len(actual_names) - len(expected_tool_names))

    args_len_ok = expected_args is None or len(expected_args) <= len(actual)
    tool_trace_ok = names_ok and args_ok and not forbidden_hits and args_len_ok

    return {
        "tool_trace_ok": tool_trace_ok,
        "tool_name_ok": names_ok,
        "tool_args_ok": args_ok,
        "forbidden_hits": forbidden_hits,
        "extra_calls": extra_calls,
        "args_detail": args_detail,
        "actual_names": actual_names,
        "expected_tool_names": list(expected_tool_names),
    }
