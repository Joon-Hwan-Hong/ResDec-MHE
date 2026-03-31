"""Tests for scripts/run_inference.py CLI behavior."""
import ast
from pathlib import Path


class TestRunInferenceScriptImports:
    """Verify run_inference.py doesn't crash on import for precomputed mode."""

    def test_no_scanpy_top_level_import(self):
        """scanpy should NOT be imported at module level — only in AnnData branch."""
        script_path = Path("scripts/inference/run_inference.py")
        tree = ast.parse(script_path.read_text())

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.append(node.module)

        assert "scanpy" not in top_level_imports, (
            "scanpy should not be a top-level import — move to AnnData branch"
        )


class TestInferenceSplitMapping:
    """Test that --split flag correctly maps to splits JSON keys."""

    def test_split_test_maps_to_holdout_test(self, tmp_path):
        """--split test should use holdout_test subjects from splits JSON."""
        splits = {
            "holdout_test": ["S1", "S2"],
            "train_val_pool": ["S3", "S4", "S5"],
            "folds": [{"train": ["S3", "S4"], "val": ["S5"]}],
            "metadata": {},
        }
        splits_path = tmp_path / "splits.json"
        import json
        splits_path.write_text(json.dumps(splits))

        from src.data.splits import load_splits, get_fold_subjects
        loaded = load_splits(splits_path)
        result = get_fold_subjects(loaded, fold_idx=0, split_type="test")
        assert result == ["S1", "S2"]

    def test_split_val_uses_fold_idx(self, tmp_path):
        """--split val should use val subjects from the specified fold."""
        splits = {
            "holdout_test": ["S1"],
            "folds": [
                {"train": ["S2", "S3"], "val": ["S4"]},
                {"train": ["S2", "S4"], "val": ["S3"]},
            ],
            "metadata": {},
        }
        splits_path = tmp_path / "splits.json"
        import json
        splits_path.write_text(json.dumps(splits))

        from src.data.splits import load_splits, get_fold_subjects
        loaded = load_splits(splits_path)
        assert get_fold_subjects(loaded, fold_idx=0, split_type="val") == ["S4"]
        assert get_fold_subjects(loaded, fold_idx=1, split_type="val") == ["S3"]


class TestRunInferenceScriptConfigRecovery:
    """Verify config can be recovered from checkpoint for data loading."""

    def test_config_recovery_codepath_exists(self):
        """Script has a config recovery path from checkpoint before failing."""
        source = (Path(__file__).resolve().parents[3] / "scripts" / "inference" / "run_inference.py").read_text()
        # The script should try predictor.config before raising ValueError
        assert "predictor.config" in source or "config = predictor" in source, (
            "run_inference.py should attempt to recover config from checkpoint "
            "before raising ValueError about missing config"
        )
