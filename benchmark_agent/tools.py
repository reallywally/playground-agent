from __future__ import annotations

import json
import math
import time
from typing import Any

from langchain_core.tools import StructuredTool


def _add_two_integers(a: int, b: int) -> int:
    """정수 a와 b의 합을 반환합니다."""
    return a + b


def _subtract_two_integers(a: int, b: int) -> int:
    """정수 a에서 b를 뺀 값을 반환합니다."""
    return a - b


def _multiply_two_integers(a: int, b: int) -> int:
    """정수 a와 b의 곱을 반환합니다."""
    return a * b


def _divide_two_integers(a: int, b: int) -> float:
    """정수 a를 b로 나눈 값을 반환합니다. b는 0이 될 수 없습니다."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b


def _int_modulo(a: int, n: int) -> int:
    """a mod n (n은 0이 아니어야 합니다)."""
    if n == 0:
        raise ValueError("modulo by zero")
    return a % n


def _abs_integer(n: int) -> int:
    """정수 n의 절댓값."""
    return abs(n)


def _min_two_integers(a: int, b: int) -> int:
    """두 정수 중 작은 값."""
    return a if a < b else b


def _max_two_integers(a: int, b: int) -> int:
    """두 정수 중 큰 값."""
    return a if a > b else b


def _is_even(n: int) -> bool:
    """n이 짝수이면 true."""
    return n % 2 == 0


def _clamp_0_100(n: int) -> int:
    """n을 0~100 범위로 클램프."""
    return max(0, min(100, n))


def _factorial_small(n: int) -> int:
    """0 이상 10 이하 정수 n의 팩토리얼."""
    if n < 0 or n > 10:
        raise ValueError("n must be between 0 and 10")
    return math.factorial(n)


def _fibonacci_small(n: int) -> int:
    """0 이상 15 이하 n에 대해 F(0)=0, F(1)=1인 피보나치."""
    if n < 0 or n > 15:
        raise ValueError("n must be between 0 and 15")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _celsius_to_fahrenheit(c: float) -> float:
    """섭씨를 화씨로 변환."""
    return c * 9 / 5 + 32


def _fahrenheit_to_celsius(f: float) -> float:
    """화씨를 섭씨로 변환."""
    return (f - 32) * 5 / 9


def _meters_to_kilometers(m: float) -> float:
    """미터를 킬로미터로."""
    return m / 1000


def _kilometers_to_meters(km: float) -> float:
    """킬로미터를 미터로."""
    return km * 1000


def _string_uppercase(s: str) -> str:
    """문자열을 대문자로."""
    return s.upper()


def _string_lowercase(s: str) -> str:
    """문자열을 소문자로."""
    return s.lower()


def _string_trim(s: str) -> str:
    """앞뒤 공백 제거."""
    return s.strip()


def _string_reverse(s: str) -> str:
    """문자열 뒤집기."""
    return s[::-1]


def _string_length(s: str) -> int:
    """문자열의 유니코드 코드포인트 개수(Python str 기준)."""
    return len(s)


def _concat_two_strings(a: str, b: str) -> str:
    """두 문자열 연결."""
    return a + b


def _substring_prefix(s: str, k: int) -> str:
    """문자열 s의 앞 k글자(k가 길이 이상이면 전체)."""
    if k <= 0:
        return ""
    return s[:k]


def _count_words_simple(s: str) -> int:
    """공백으로 분리한 단어 수(연속 공백은 split() 규칙에 따름)."""
    return len(s.split())


def _hex_string_to_int(hex_string: str) -> int:
    """16진 문자열을 정수로(0x 접두 허용)."""
    text = hex_string.strip()
    if text.startswith("0x") or text.startswith("0X"):
        text = text[2:]
    return int(text, 16)


def _int_to_hex_string(n: int) -> str:
    """정수 n을 소문자 16진 문자열로(음수는 - 접두 + 절댓값 16진)."""
    if n < 0:
        return "-" + format(abs(n), "x")
    return format(n, "x")


def _parse_json_keys(json_string: str) -> list[str]:
    """JSON 문자열의 최상위 키 목록(정렬)."""
    data = json.loads(json_string)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return sorted(data.keys())


def _sort_two_strings(a: str, b: str) -> list[str]:
    """두 문자열을 사전순 정렬한 길이 2 리스트."""
    return sorted([a, b])


def _unix_timestamp_seconds() -> int:
    """현재 Unix epoch 초."""
    return int(time.time())


def _noop_identity(s: str) -> str:
    """입력 문자열을 그대로 반환(노이즈·오버헤드 측정용)."""
    return s


_TOOL_SPECS: list[tuple[str, Any, str]] = [
    ("add_two_integers", _add_two_integers, _add_two_integers.__doc__ or ""),
    ("subtract_two_integers", _subtract_two_integers, _subtract_two_integers.__doc__ or ""),
    ("multiply_two_integers", _multiply_two_integers, _multiply_two_integers.__doc__ or ""),
    ("divide_two_integers", _divide_two_integers, _divide_two_integers.__doc__ or ""),
    ("int_modulo", _int_modulo, _int_modulo.__doc__ or ""),
    ("abs_integer", _abs_integer, _abs_integer.__doc__ or ""),
    ("min_two_integers", _min_two_integers, _min_two_integers.__doc__ or ""),
    ("max_two_integers", _max_two_integers, _max_two_integers.__doc__ or ""),
    ("is_even", _is_even, _is_even.__doc__ or ""),
    ("clamp_0_100", _clamp_0_100, _clamp_0_100.__doc__ or ""),
    ("factorial_small", _factorial_small, _factorial_small.__doc__ or ""),
    ("fibonacci_small", _fibonacci_small, _fibonacci_small.__doc__ or ""),
    ("celsius_to_fahrenheit", _celsius_to_fahrenheit, _celsius_to_fahrenheit.__doc__ or ""),
    ("fahrenheit_to_celsius", _fahrenheit_to_celsius, _fahrenheit_to_celsius.__doc__ or ""),
    ("meters_to_kilometers", _meters_to_kilometers, _meters_to_kilometers.__doc__ or ""),
    ("kilometers_to_meters", _kilometers_to_meters, _kilometers_to_meters.__doc__ or ""),
    ("string_uppercase", _string_uppercase, _string_uppercase.__doc__ or ""),
    ("string_lowercase", _string_lowercase, _string_lowercase.__doc__ or ""),
    ("string_trim", _string_trim, _string_trim.__doc__ or ""),
    ("string_reverse", _string_reverse, _string_reverse.__doc__ or ""),
    ("string_length", _string_length, _string_length.__doc__ or ""),
    ("concat_two_strings", _concat_two_strings, _concat_two_strings.__doc__ or ""),
    ("substring_prefix", _substring_prefix, _substring_prefix.__doc__ or ""),
    ("count_words_simple", _count_words_simple, _count_words_simple.__doc__ or ""),
    ("hex_string_to_int", _hex_string_to_int, _hex_string_to_int.__doc__ or ""),
    ("int_to_hex_string", _int_to_hex_string, _int_to_hex_string.__doc__ or ""),
    ("parse_json_keys", _parse_json_keys, _parse_json_keys.__doc__ or ""),
    ("sort_two_strings", _sort_two_strings, _sort_two_strings.__doc__ or ""),
    ("unix_timestamp_seconds", _unix_timestamp_seconds, _unix_timestamp_seconds.__doc__ or ""),
    ("noop_identity", _noop_identity, _noop_identity.__doc__ or ""),
]


def all_tool_names() -> list[str]:
    return [name for name, _, _ in _TOOL_SPECS]


def build_tool_catalog() -> dict[str, StructuredTool]:
    """이름 → LangChain StructuredTool (카탈로그 전체)."""
    catalog: dict[str, StructuredTool] = {}
    for name, fn, desc in _TOOL_SPECS:
        catalog[name] = StructuredTool.from_function(fn, name=name, description=desc.strip())
    return catalog


def tools_for_active_names(
    active_tool_names: list[str],
    *,
    catalog: dict[str, StructuredTool] | None = None,
) -> list[StructuredTool]:
    """활성 이름 순서대로 툴 리스트를 구성합니다."""
    cat = catalog or build_tool_catalog()
    out: list[StructuredTool] = []
    for name in active_tool_names:
        if name not in cat:
            raise ValueError(f"Unknown tool name: {name}")
        out.append(cat[name])
    return out


def tool_result_to_content(value: Any) -> str:
    """ToolMessage content로 직렬화."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
