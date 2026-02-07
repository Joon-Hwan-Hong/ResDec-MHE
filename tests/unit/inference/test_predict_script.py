"""Tests for scripts/run_inference.py CLI behavior."""
import ast
from pathlib import Path


class TestRunInferenceScriptImports:
    """Verify run_inference.py doesn't crash on import for precomputed mode."""

    def test_no_scanpy_top_level_import(self):
        """scanpy should NOT be imported at module level — only in AnnData branch."""
        script_path = Path("scripts/run_inference.py")
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


class TestRunInferenceScriptConfigRecovery:
    """Verify config can be recovered from checkpoint for data loading."""

    def test_config_recovery_from_checkpoint(self):
        """When --config is omitted, script should attempt predictor.config before failing."""
        source = Path("scripts/run_inference.py").read_text()
        assert 'Config is required for data loading' not in source, (
            "Hard-fail for missing config should be replaced with checkpoint config recovery"
        )
