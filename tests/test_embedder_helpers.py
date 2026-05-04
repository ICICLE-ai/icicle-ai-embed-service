"""Tests for pure helper functions in embedder.py.

These touch no model and no network — fast and deterministic.
"""

import math

from src.app.embedder import (
    DEFAULT_QUERY_INSTRUCTION,
    _format_query,
    _l2_normalize,
)


def test_format_query_uses_default_instruction():
    out = _format_query("how do plants make food", None)
    assert DEFAULT_QUERY_INSTRUCTION in out
    assert "Query: how do plants make food" in out


def test_format_query_uses_custom_instruction():
    out = _format_query("hello", "Find related code")
    assert "Find related code" in out
    assert DEFAULT_QUERY_INSTRUCTION not in out


def test_format_query_template_shape():
    out = _format_query("foo", "Bar")
    assert out.startswith("Instruct: Bar\nQuery: foo")


def test_l2_normalize_produces_unit_vector():
    out = _l2_normalize([3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in out))
    assert abs(norm - 1.0) < 1e-9


def test_l2_normalize_preserves_direction():
    v = [1.0, 2.0, 3.0]
    out = _l2_normalize(v)
    # Same direction => same ratios between components
    assert abs(out[0] / out[1] - v[0] / v[1]) < 1e-9
    assert abs(out[1] / out[2] - v[1] / v[2]) < 1e-9


def test_l2_normalize_handles_zero_vector():
    # Should not divide by zero — returns input unchanged.
    assert _l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_l2_normalize_handles_negative_components():
    out = _l2_normalize([-3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in out))
    assert abs(norm - 1.0) < 1e-9
    assert out[0] < 0
    assert out[1] > 0


def test_l2_normalize_idempotent_on_unit_vector():
    v = [0.6, 0.8]  # already unit length
    out = _l2_normalize(v)
    assert all(abs(a - b) < 1e-9 for a, b in zip(v, out))
