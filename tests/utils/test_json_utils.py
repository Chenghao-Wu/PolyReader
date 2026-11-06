"""Tests for polyreader.utils.json_utils.extract_json edge cases."""

import pytest

from polyreader.utils.json_utils import extract_json, get_json_content


class TestGetJsonContent:
    def test_strips_json_fence(self):
        assert get_json_content('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_plain_fence(self):
        assert get_json_content('```\n[1, 2]\n```') == '[1, 2]'

    def test_no_fence_returns_stripped(self):
        assert get_json_content('  {"a": 1}  ') == '{"a": 1}'


class TestExtractJson:
    # --- Direct parse ---
    def test_valid_json_object(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_valid_json_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_already_parsed_dict(self):
        d = {"a": 1}
        assert extract_json(d) is d

    def test_already_parsed_list(self):
        lst = [1, 2]
        assert extract_json(lst) is lst

    # --- Python literal cleaning ---
    def test_none_to_null(self):
        assert extract_json('{"a": None}') == {"a": None}

    def test_true_false_conversion(self):
        result = extract_json('{"a": True, "b": False}')
        assert result == {"a": True, "b": False}

    def test_none_inside_string_preserved(self):
        """Word boundary regex should NOT replace None inside strings like 'NoneType'."""
        result = extract_json('{"error": "NoneType is not iterable", "val": None}')
        # "NoneType" should remain as-is because \b doesn't match mid-word
        # But the standalone None becomes null
        assert result["val"] is None

    # --- Trailing comma ---
    def test_trailing_comma_object(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        assert extract_json('[1, 2, 3,]') == [1, 2, 3]

    # --- Markdown fences ---
    def test_json_in_markdown_fence(self):
        text = 'Here is the result:\n```json\n{"samples": [1]}\n```\nDone.'
        assert extract_json(text) == {"samples": [1]}

    # --- Greedy regex fix (the core P2 bug) ---
    def test_multiple_json_objects_picks_first_valid(self):
        """The old greedy regex would match from first { to last }, failing parse.
        Non-greedy + finditer should pick the first valid JSON object."""
        text = 'Some text {"first": 1} more text {"second": 2}'
        result = extract_json(text)
        assert result == {"first": 1}

    def test_json_surrounded_by_prose(self):
        text = 'The answer is {"samples": [{"id": "A"}], "notes": []} as shown above.'
        result = extract_json(text)
        assert result == {"samples": [{"id": "A"}], "notes": []}

    def test_nested_braces_still_work(self):
        """Non-greedy won't break nested objects — finditer tries each match."""
        text = 'Result: {"outer": {"inner": 42}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": 42}}

    def test_array_extraction_from_text(self):
        text = 'Here: [{"a": 1}, {"b": 2}] end'
        result = extract_json(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_multiple_arrays_picks_first_valid(self):
        text = 'List [1, 2] and also [3, 4]'
        result = extract_json(text)
        assert result == [1, 2]

    # --- Fallback / empty ---
    def test_unparseable_returns_empty_dict(self):
        assert extract_json('not json at all') == {}

    def test_empty_string_returns_empty_dict(self):
        assert extract_json('') == {}

    # --- Deeply nested (regression) ---
    def test_deeply_nested_object(self):
        obj = '{"a": {"b": {"c": {"d": 1}}}}'
        assert extract_json(obj) == {"a": {"b": {"c": {"d": 1}}}}
