"""Unit tests for src/utils/device.py — move_batch_to_device and helpers."""

import pytest
import torch

from src.utils.device import move_batch_to_device, _move_to_device


class TestMoveToDevice:
    """Tests for the recursive _move_to_device helper."""

    def test_tensor_moves_to_device(self):
        t = torch.randn(3, 4)
        moved = _move_to_device(t, torch.device("cpu"))
        assert moved.device == torch.device("cpu")

    def test_nested_dict_moves_tensors(self):
        data = {"a": torch.randn(2), "b": {"c": torch.randn(3)}}
        moved = _move_to_device(data, torch.device("cpu"))
        assert isinstance(moved["b"], dict)
        assert isinstance(moved["b"]["c"], torch.Tensor)

    def test_nested_list_moves_tensors(self):
        data = [torch.randn(2), torch.randn(3)]
        moved = _move_to_device(data, torch.device("cpu"))
        assert len(moved) == 2
        assert all(isinstance(t, torch.Tensor) for t in moved)

    def test_non_tensor_passthrough(self):
        assert _move_to_device("hello", torch.device("cpu")) == "hello"
        assert _move_to_device(42, torch.device("cpu")) == 42
        assert _move_to_device(None, torch.device("cpu")) is None

    def test_empty_structures(self):
        assert _move_to_device({}, torch.device("cpu")) == {}
        assert _move_to_device([], torch.device("cpu")) == []

    def test_tuple_keys_preserved_in_dict(self):
        data = {("TypeA", "rel", "TypeB"): torch.randn(2, 3)}
        moved = _move_to_device(data, torch.device("cpu"))
        assert ("TypeA", "rel", "TypeB") in moved

    def test_dtype_preserved(self):
        for dtype in [torch.float32, torch.float64, torch.bool, torch.int64]:
            t = torch.zeros(2, dtype=dtype)
            moved = _move_to_device(t, torch.device("cpu"))
            assert moved.dtype == dtype

    def test_scalar_tensor(self):
        t = torch.tensor(3.14)
        moved = _move_to_device(t, torch.device("cpu"))
        assert moved.item() == pytest.approx(3.14)

    def test_empty_tensor(self):
        t = torch.empty(0, 3)
        moved = _move_to_device(t, torch.device("cpu"))
        assert moved.shape == (0, 3)


class TestMoveBatchToDevice:
    """Tests for move_batch_to_device."""

    def test_string_device_accepted(self):
        batch = {"x": torch.randn(2, 3)}
        moved = move_batch_to_device(batch, "cpu")
        assert moved["x"].device == torch.device("cpu")

    def test_torch_device_accepted(self):
        batch = {"x": torch.randn(2, 3)}
        moved = move_batch_to_device(batch, torch.device("cpu"))
        assert moved["x"].device == torch.device("cpu")

    def test_cpu_keys_preserved(self):
        batch = {
            "pseudobulk": torch.randn(4, 31, 50),
            "subject_ids": ["subj_0", "subj_1", "subj_2", "subj_3"],
            "batch_size": 4,
            "n_nodes_per_graph": [31, 31, 31, 31],
        }
        moved = move_batch_to_device(batch, "cpu")
        assert moved["subject_ids"] is batch["subject_ids"]
        assert moved["batch_size"] is batch["batch_size"]
        assert moved["n_nodes_per_graph"] is batch["n_nodes_per_graph"]

    def test_raw_edge_tensors_moved_to_device(self):
        batch = {
            "ccc_edge_index": torch.tensor([[[0], [0]], [[0], [0]]]),
            "ccc_edge_type": torch.tensor([[0], [0]]),
            "ccc_edge_attr": torch.tensor([[[0.5]], [[0.5]]]),
            "ccc_edge_counts": torch.tensor([1, 1]),
        }
        moved = move_batch_to_device(batch, "cpu")
        for key in ("ccc_edge_index", "ccc_edge_type", "ccc_edge_attr", "ccc_edge_counts"):
            assert isinstance(moved[key], torch.Tensor)
            assert moved[key].device == torch.device("cpu")

    def test_mixed_batch_all_keys_present(self):
        batch = {
            "pseudobulk": torch.randn(4, 31, 50),
            "cells": torch.randn(4, 31, 100, 50),
            "cell_mask": torch.ones(4, 31, 100, dtype=torch.bool),
            "cognition": torch.randn(4, 1),
            "pathology": torch.randn(4, 3),
            "region_pseudobulk": torch.randn(4, 6, 31, 50),
            "region_mask": torch.ones(4, 6, dtype=torch.bool),
            "ccc_edge_index": torch.zeros(4, 2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(4, 0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(4, 0, 1),
            "ccc_edge_counts": torch.zeros(4, dtype=torch.long),
            "subject_ids": [f"subj_{i}" for i in range(4)],
            "batch_size": 4,
        }
        moved = move_batch_to_device(batch, "cpu")
        assert set(moved.keys()) == set(batch.keys())
        assert isinstance(moved["pseudobulk"], torch.Tensor)
        assert moved["subject_ids"] == batch["subject_ids"]

    def test_none_values_in_batch(self):
        batch = {"x": torch.randn(2), "optional": None}
        moved = move_batch_to_device(batch, "cpu")
        assert moved["optional"] is None

    def test_empty_edge_tensors(self):
        batch = {
            "ccc_edge_index": torch.zeros(3, 2, 0, dtype=torch.long),
            "ccc_edge_type": torch.zeros(3, 0, dtype=torch.long),
            "ccc_edge_attr": torch.zeros(3, 0, 1),
            "ccc_edge_counts": torch.zeros(3, dtype=torch.long),
        }
        moved = move_batch_to_device(batch, "cpu")
        assert (moved["ccc_edge_counts"] == 0).all()
