"""Tests for ``src.utils.cell_types.pad_cell_type_names``."""
from __future__ import annotations

import pytest

from src.utils.cell_types import pad_cell_type_names


class TestPadCellTypeNames:
    """Cover the truncate / pass-through / pad / edge-case branches."""

    def test_truncates_when_source_longer(self):
        names = ["A", "B", "C", "D", "E"]
        assert pad_cell_type_names(names, 3) == ["A", "B", "C"]

    def test_passes_through_when_lengths_match(self):
        names = ["A", "B", "C"]
        assert pad_cell_type_names(names, 3) == ["A", "B", "C"]

    def test_pads_with_default_prefix(self):
        names = ["A", "B"]
        assert pad_cell_type_names(names, 4) == ["A", "B", "ct_2", "ct_3"]

    def test_pads_with_custom_prefix(self):
        names = ["A"]
        assert pad_cell_type_names(names, 3, prefix="unknown_") == [
            "A", "unknown_1", "unknown_2",
        ]

    def test_empty_source_pads_full_length(self):
        assert pad_cell_type_names([], 3) == ["ct_0", "ct_1", "ct_2"]

    def test_zero_length_returns_empty(self):
        assert pad_cell_type_names(["A", "B"], 0) == []

    def test_negative_n_ct_raises(self):
        with pytest.raises(ValueError, match="n_ct must be non-negative"):
            pad_cell_type_names(["A"], -1)

    def test_accepts_iterable_not_just_list(self):
        # tuple input
        assert pad_cell_type_names(("A", "B"), 3) == ["A", "B", "ct_2"]
        # generator input
        gen = (f"x{i}" for i in range(3))
        assert pad_cell_type_names(gen, 5) == ["x0", "x1", "x2", "ct_3", "ct_4"]

    def test_does_not_mutate_input(self):
        names = ["A", "B"]
        pad_cell_type_names(names, 4)
        assert names == ["A", "B"]  # source unchanged

    def test_returns_new_list(self):
        names = ["A", "B", "C"]
        out = pad_cell_type_names(names, 3)
        out.append("Z")
        assert names == ["A", "B", "C"]  # original list intact

    def test_canonical_n_ct_31_pad_branch(self):
        """Real-world case: source is shorter than the model's n_ct=31."""
        out = pad_cell_type_names(["Astrocyte", "Microglia"], 31)
        assert len(out) == 31
        assert out[0] == "Astrocyte"
        assert out[1] == "Microglia"
        assert out[2] == "ct_2"
        assert out[30] == "ct_30"
