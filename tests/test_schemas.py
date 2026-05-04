"""Pydantic validator tests for EmbedRequest.

Highest-ROI tests: every input-validation rule we care about lives here, and
they run in milliseconds without any I/O.
"""

import pytest
from pydantic import ValidationError

from src.app.schemas import EmbedRequest


def test_single_string_accepted():
    req = EmbedRequest(input="hello")
    assert req.input == "hello"


def test_list_accepted():
    req = EmbedRequest(input=["a", "b", "c"])
    assert req.input == ["a", "b", "c"]


def test_empty_string_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        EmbedRequest(input="")


def test_whitespace_only_string_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        EmbedRequest(input="    \t\n")


def test_empty_list_rejected():
    with pytest.raises(ValidationError, match="at least one"):
        EmbedRequest(input=[])


def test_list_with_empty_item_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        EmbedRequest(input=["good", ""])


def test_list_with_whitespace_item_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        EmbedRequest(input=["good", "   "])


def test_oversized_string_rejected():
    with pytest.raises(ValidationError, match="max_chars"):
        EmbedRequest(input="x" * 200_001)


def test_oversized_list_rejected():
    with pytest.raises(ValidationError, match="max_inputs"):
        EmbedRequest(input=["x"] * 257)


def test_default_input_type_is_document():
    req = EmbedRequest(input="hello")
    assert req.input_type == "document"


def test_default_normalize_is_true():
    req = EmbedRequest(input="hello")
    assert req.normalize is True


def test_default_instruction_is_none():
    req = EmbedRequest(input="hello")
    assert req.instruction is None


def test_invalid_input_type_rejected():
    with pytest.raises(ValidationError):
        EmbedRequest(input="hello", input_type="something_else")  # type: ignore[arg-type]


def test_query_input_type_accepted():
    req = EmbedRequest(input="hello", input_type="query")
    assert req.input_type == "query"


def test_oversized_instruction_rejected():
    with pytest.raises(ValidationError):
        EmbedRequest(input="hello", instruction="x" * 2001)


def test_normalize_can_be_false():
    req = EmbedRequest(input="hello", normalize=False)
    assert req.normalize is False
