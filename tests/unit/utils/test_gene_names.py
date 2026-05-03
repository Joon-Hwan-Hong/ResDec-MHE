"""Tests for ``src.utils.gene_names.load_gene_names``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.gene_names import load_gene_names


class TestLoadGeneNames:
    """Cover .npy load / .json load / fallback / placeholder branches."""

    def test_loads_from_npy(self, tmp_path: Path):
        names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
        np.save(tmp_path / "gene_names.npy", np.asarray(names, dtype=object))
        out, used_real = load_gene_names(tmp_path, n_genes=3)
        assert out == ["GENE_A", "GENE_B", "GENE_C"]
        assert used_real is True

    def test_loads_from_json(self, tmp_path: Path):
        names = ["FOO", "BAR", "BAZ"]
        (tmp_path / "gene_names.json").write_text(json.dumps(names))
        out, used_real = load_gene_names(tmp_path, n_genes=3)
        assert out == ["FOO", "BAR", "BAZ"]
        assert used_real is True

    def test_loads_from_feature_names_json(self, tmp_path: Path):
        # Sidecar named feature_names.json is also a recognised candidate.
        names = ["X1", "X2", "X3"]
        (tmp_path / "feature_names.json").write_text(json.dumps(names))
        out, used_real = load_gene_names(tmp_path, n_genes=2)
        assert out == ["X1", "X2"]
        assert used_real is True

    def test_falls_back_to_placeholders_when_missing(self, tmp_path: Path):
        # No sidecar in the tmp dir; fallback_paths empty so no file at all.
        out, used_real = load_gene_names(
            tmp_path,
            n_genes=4,
            fallback_paths=(),
        )
        assert out == ["gene_0", "gene_1", "gene_2", "gene_3"]
        assert used_real is False

    def test_falls_back_when_file_too_short(self, tmp_path: Path):
        # File exists but has fewer entries than n_genes.
        (tmp_path / "gene_names.json").write_text(json.dumps(["A", "B"]))
        out, used_real = load_gene_names(
            tmp_path,
            n_genes=5,
            fallback_paths=(),
        )
        # Falls through to placeholders.
        assert out == ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]
        assert used_real is False

    def test_npy_priority_over_json(self, tmp_path: Path):
        # gene_names.npy is the first candidate; should win over gene_names.json.
        np.save(
            tmp_path / "gene_names.npy",
            np.asarray(["NPY_1", "NPY_2"], dtype=object),
        )
        (tmp_path / "gene_names.json").write_text(json.dumps(["JSON_1", "JSON_2"]))
        out, used_real = load_gene_names(tmp_path, n_genes=2)
        assert out == ["NPY_1", "NPY_2"]
        assert used_real is True

    def test_fallback_path_used_when_primary_missing(self, tmp_path: Path):
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()
        fallback_file = fallback_dir / "gene_names.json"
        fallback_file.write_text(json.dumps(["FB_A", "FB_B", "FB_C"]))

        # Empty primary dir, valid fallback.
        primary = tmp_path / "primary"
        primary.mkdir()
        out, used_real = load_gene_names(
            primary,
            n_genes=2,
            fallback_paths=(fallback_file,),
        )
        assert out == ["FB_A", "FB_B"]
        assert used_real is True

    def test_zero_n_genes_returns_empty(self, tmp_path: Path):
        out, used_real = load_gene_names(
            tmp_path, n_genes=0, fallback_paths=(),
        )
        assert out == []
        # An empty file-name list is still always >= 0 entries; placeholder branch
        # since no sidecar was found.
        assert used_real is False

    def test_returns_strings_even_for_numeric_npy(self, tmp_path: Path):
        # Non-string entries in the .npy still get cast to str.
        np.save(tmp_path / "gene_names.npy", np.asarray([1, 2, 3]))
        out, used_real = load_gene_names(tmp_path, n_genes=2)
        assert out == ["1", "2"]
        assert used_real is True
