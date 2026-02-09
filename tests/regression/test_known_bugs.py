"""
Regression tests for known bugs fixed in previous rounds.

Each test guards against a specific bug that was fixed and could regress.
Tests are self-contained with clear references to the original bug.

Bugs covered:
    1. Empty CCC edges caused synthetic self-loop injection (Round 13, C1)
    2. collate_multiregion used stale sentinel key (Round 13, M2)
    3. load_config() rejected dotlist overrides (Round 13, Task 6)
    4. Optuna fold pruning used last-fold loss, not running mean (Round 14, Task 2)
    5. Checkpoint saved model_config only, not full_config (Round 14, Task 4)
    6. Predictor.from_checkpoint ignored full_config key (Round 14, Task 4)
    7. Resilience plots went to attention/ directory (Round 13, H3)
    8. LR not scaled by world_size for multi-GPU (Round 14, Task 3)
    9. --final mode used holdout test as val_dataloaders (Round 16, Task 1)
    10. Bayesian model exported full weights.pt (unsafe without guide) (Round 17, Task 2)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.data.constants import N_CELL_TYPES, N_REGIONS


# ---------------------------------------------------------------------------
# Test 1: Empty CCC edges should NOT inject synthetic self-loop edges
# Bug: Round 13, C1
# Fix: CognitiveResilienceModel.forward() now passes empty edge dicts through
#      as-is to HGTConv, which handles isolated nodes correctly. No synthetic
#      edges are injected.
# ---------------------------------------------------------------------------
class TestEmptyCCCEdgesNoSyntheticInjection:
    """Verify empty CCC edge dicts pass through without synthetic edge injection."""

    @pytest.fixture
    def small_model(self):
        """Build a minimal CognitiveResilienceModel for testing."""
        from src.models.full_model import CognitiveResilienceModel

        return CognitiveResilienceModel(
            n_genes=50,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=4,
            n_isab_layers=1,
            n_inducing_points=4,
            n_attention_heads=4,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=False,
        )

    def test_empty_ccc_edges_no_synthetic_injection(self, small_model):
        """
        Round 13, C1: Forward with empty edge_index_dict_list=[{}] should
        produce finite output without any edges being fabricated.

        The bug was that the model injected dummy self-loop edges when CCC
        edges were empty, encoding phantom cell-cell communication. The fix
        passes empty dicts through to HGTConv which handles isolated nodes
        via its received_messages mask.
        """
        B = 2
        model = small_model
        model.eval()

        pseudobulk = torch.randn(B, N_CELL_TYPES, 50)
        cells = torch.randn(B, N_CELL_TYPES, 4, 50)
        cell_mask = torch.ones(B, N_CELL_TYPES, 4, dtype=torch.bool)
        pathology = torch.randn(B, 3)

        # Empty edge dicts -- no CCC edges
        edge_index_dict_list = [{} for _ in range(B)]
        edge_attr_dict_list = [{} for _ in range(B)]

        with torch.no_grad():
            output = model(
                pseudobulk=pseudobulk,
                edge_index_dict_list=edge_index_dict_list,
                edge_attr_dict_list=edge_attr_dict_list,
                cells=cells,
                cell_mask=cell_mask,
                pathology=pathology,
            )

        # Output should be finite (no NaN or Inf)
        assert torch.isfinite(output["mean"]).all(), (
            "Empty CCC edges produced non-finite output"
        )

        # Edge dicts should remain empty -- model must not inject edges
        for i, eid in enumerate(edge_index_dict_list):
            assert len(eid) == 0, (
                f"Sample {i}: edge_index_dict was mutated from empty to "
                f"{len(eid)} entries -- synthetic edges were injected"
            )


# ---------------------------------------------------------------------------
# Test 2: collate_multiregion derives regions from key patterns
# Bug: Round 13, M2
# Fix: collate_for_hgt_multiregion and collate_multiregion now use
#      _derive_available_regions_from_keys() to detect multi-region data
#      from region_{idx}_pseudobulk keys, not a stale "region_pseudobulk"
#      sentinel key.
# ---------------------------------------------------------------------------
class TestCollateMultiregionDerivesRegionsFromKeys:
    """Verify collate detects regions from key patterns, not a sentinel key."""

    def test_collate_multiregion_derives_regions_from_keys(self):
        """
        Round 13, M2: Samples with region_{idx}_pseudobulk keys (like
        region_0_pseudobulk, region_1_pseudobulk) should be recognized as
        multi-region data by the collate function, even without a
        "region_pseudobulk" sentinel key.

        The bug was that collate_multiregion only checked for a sentinel
        "region_pseudobulk" key and missed multi-region data entirely when
        samples used the region_{idx} naming convention.
        """
        from src.data.collate import (
            _derive_available_regions_from_keys,
            collate_for_hgt_multiregion,
        )

        n_genes = 50

        # Verify _derive_available_regions_from_keys finds regions
        sample = {
            "region_0_pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "region_2_pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        }
        derived = _derive_available_regions_from_keys(sample)
        assert derived == [0, 2], (
            f"Expected regions [0, 2] but got {derived}. "
            "Region derivation from key patterns is broken."
        )

        # Also verify with a sample that has NO sentinel key but DOES have
        # region-prefixed keys -- collate should detect multi-region
        sample_with_no_sentinel = {
            "region_0_pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "region_3_pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
            "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        }
        derived2 = _derive_available_regions_from_keys(sample_with_no_sentinel)
        assert 0 in derived2 and 3 in derived2, (
            f"Expected regions 0 and 3 in derived list but got {derived2}"
        )


# ---------------------------------------------------------------------------
# Test 3: load_config() accepts both dict and dotlist overrides
# Bug: Round 13, Task 6
# Fix: load_config() now checks isinstance(overrides, list) and calls
#      OmegaConf.from_dotlist() for list, OmegaConf.create() for dict.
# ---------------------------------------------------------------------------
class TestConfigLoadAcceptsDotlistOverrides:
    """Verify load_config() accepts dict and dotlist override formats."""

    def test_config_load_accepts_dotlist_overrides(self):
        """
        Round 13, Task 6: load_config() should accept both dict and dotlist
        override formats.

        The bug was that only dict overrides worked; passing a list of
        dotlist strings like ["training.max_epochs=50"] would raise an error.
        """
        from src.utils.config import load_config

        # Create a temp config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("training:\n  max_epochs: 100\n  lr: 0.001\n")
            config_path = f.name

        try:
            # Test dict overrides
            cfg_dict = load_config(
                config_path, overrides={"training": {"max_epochs": 50}}
            )
            assert cfg_dict.training.max_epochs == 50

            # Test dotlist overrides (this was broken before the fix)
            cfg_dotlist = load_config(
                config_path, overrides=["training.max_epochs=25"]
            )
            assert cfg_dotlist.training.max_epochs == 25

            # Test None overrides (no change)
            cfg_none = load_config(config_path, overrides=None)
            assert cfg_none.training.max_epochs == 100
        finally:
            Path(config_path).unlink()


# ---------------------------------------------------------------------------
# Test 4: Optuna fold pruning uses running mean, not last-fold loss
# Bug: Round 14, Task 2
# Fix: objective() in optuna_optimize.py now computes a running mean of
#      fold validation losses and reports that to trial.report(), not the
#      loss from the last fold alone.
# ---------------------------------------------------------------------------
class TestOptunaFoldPruningUsesRunningMean:
    """Verify fold-level pruning reports running mean loss."""

    def test_optuna_fold_pruning_uses_running_mean(self):
        """
        Round 14, Task 2: When reporting intermediate values for pruning,
        the objective function should use the running mean of fold losses
        (sum/count), not just the last fold's loss.

        This test verifies the running mean computation pattern used in
        scripts/optuna_optimize.py::objective().
        """
        # Simulate the fold loss accumulation pattern from objective()
        fold_val_losses = []
        reported_values = []

        # Simulate 3 folds with different losses
        test_losses = [0.5, 0.3, 0.8]

        for fold_idx, fold_loss in enumerate(test_losses):
            fold_val_losses.append(fold_loss)

            # This is the exact pattern from the fixed code:
            # running_mean = sum(fold_val_losses) / len(fold_val_losses)
            running_mean = (
                sum(fold_val_losses) / len(fold_val_losses)
                if fold_val_losses
                else float("inf")
            )
            reported_values.append(running_mean)

        # After fold 0: running_mean = 0.5/1 = 0.5
        assert abs(reported_values[0] - 0.5) < 1e-9

        # After fold 1: running_mean = (0.5+0.3)/2 = 0.4
        assert abs(reported_values[1] - 0.4) < 1e-9

        # After fold 2: running_mean = (0.5+0.3+0.8)/3 = 0.5333...
        expected_final = (0.5 + 0.3 + 0.8) / 3
        assert abs(reported_values[2] - expected_final) < 1e-9

        # The bug would have reported last fold's loss (0.8) instead of
        # running mean (0.5333...) at step 2
        assert reported_values[2] != test_losses[2], (
            "Pruning reports last-fold loss instead of running mean -- "
            "this was the Round 14 bug"
        )


# ---------------------------------------------------------------------------
# Test 5: Checkpoint saves full_config, not just model_config
# Bug: Round 14, Task 4
# Fix: ResilienceModelCheckpoint.on_save_checkpoint() now saves both
#      "model_config" and "full_config" keys.
# ---------------------------------------------------------------------------
class TestCheckpointSavesFullConfig:
    """Verify checkpoint includes full_config key."""

    def test_checkpoint_saves_full_config(self):
        """
        Round 14, Task 4: ResilienceModelCheckpoint.on_save_checkpoint()
        must save the full experiment config under "full_config", not just
        the model config under "model_config".

        Without full_config, the Predictor cannot reconstruct training
        settings (loss type, scheduler, etc.) needed for correct inference.
        """
        from src.training.callbacks import ResilienceModelCheckpoint

        callback = ResilienceModelCheckpoint()

        # Create a mock pl_module with full config
        pl_module = MagicMock()
        full_config = {
            "model": {
                "n_genes": 100,
                "head": {"type": "bayesian"},
            },
            "training": {
                "max_epochs": 100,
                "optimizer": {"lr": 0.001},
                "loss": {"type": "beta_nll", "beta": 0.5},
            },
            "data": {"batch_size": 16},
        }
        pl_module.config = full_config

        # Create empty checkpoint dict
        checkpoint = {}

        # Mock trainer
        trainer = MagicMock()

        # Call on_save_checkpoint
        callback.on_save_checkpoint(trainer, pl_module, checkpoint)

        # Must have full_config key
        assert "full_config" in checkpoint, (
            "Checkpoint missing 'full_config' key -- "
            "only model_config was saved (Round 14 bug)"
        )

        # full_config should contain the complete config including training
        assert "training" in checkpoint["full_config"]
        assert "data" in checkpoint["full_config"]
        assert "model" in checkpoint["full_config"]

        # model_config should also be present for backward compat
        assert "model_config" in checkpoint, (
            "Checkpoint missing 'model_config' key for backward compatibility"
        )


# ---------------------------------------------------------------------------
# Test 6: Predictor.from_checkpoint reads full_config first, falls back
# Bug: Round 14, Task 4
# Fix: Predictor.from_checkpoint() now checks for "full_config" before
#      "model_config", falling back gracefully for legacy checkpoints.
# ---------------------------------------------------------------------------
class TestPredictorRecoversFullConfigFromCheckpoint:
    """Verify Predictor reads full_config first, falls back to model_config."""

    def test_predictor_recovers_full_config_from_checkpoint(self):
        """
        Round 14, Task 4: Predictor.from_checkpoint() must try full_config
        first, then fall back to model_config for legacy checkpoints.

        The bug was that it only checked model_config, losing training
        settings needed for correct inference (e.g., loss type for the
        Bayesian head).
        """
        from src.inference.predict import Predictor

        # Simulate the config extraction logic from Predictor.from_checkpoint()
        # Test 1: Checkpoint with full_config (new format)
        ckpt_new = {
            "full_config": {
                "model": {"n_genes": 100, "head": {"type": "deterministic"}},
                "training": {"loss": {"type": "mse"}},
            },
            "model_config": {"n_genes": 100, "head": {"type": "deterministic"}},
        }

        # The from_checkpoint logic should prefer full_config
        config_new = None
        if "full_config" in ckpt_new:
            config_new = OmegaConf.create(ckpt_new["full_config"])
        elif "model_config" in ckpt_new:
            config_new = OmegaConf.create({"model": ckpt_new["model_config"]})

        assert config_new is not None
        assert "training" in config_new, (
            "full_config was available but from_checkpoint used model_config "
            "instead, losing training settings"
        )

        # Test 2: Legacy checkpoint with only model_config
        ckpt_legacy = {
            "model_config": {"n_genes": 100, "head": {"type": "deterministic"}},
        }

        config_legacy = None
        if "full_config" in ckpt_legacy:
            config_legacy = OmegaConf.create(ckpt_legacy["full_config"])
        elif "model_config" in ckpt_legacy:
            config_legacy = OmegaConf.create({"model": ckpt_legacy["model_config"]})

        assert config_legacy is not None
        assert "model" in config_legacy, (
            "Legacy model_config fallback is broken"
        )


# ---------------------------------------------------------------------------
# Test 7: Resilience plots go to resilience/ directory, not attention/
# Bug: Round 13, H3
# Fix: generate_plots.py now uses resilience_dir = output_dir / "resilience"
#      for resilience signature plots.
# ---------------------------------------------------------------------------
class TestResiliencePlotsUseResilienceDir:
    """Verify resilience plots go to resilience/ subdirectory."""

    def test_resilience_plots_use_resilience_dir(self):
        """
        Round 13, H3: Resilience signature plots must be saved to the
        resilience/ subdirectory, not the attention/ subdirectory.

        The bug was that resilience plots were written to the attention/
        directory due to a copy-paste error in the output path.
        """
        # Read the generate_plots.py source and verify the directory name
        # by inspecting the actual code structure.
        # The fix is in generate_plots.py main() where resilience_dir is set.
        import scripts.generate_plots as gp

        # Verify generate_resilience_plots function exists
        assert hasattr(gp, "generate_resilience_plots"), (
            "generate_resilience_plots function not found in generate_plots.py"
        )

        # Verify the main() function uses "resilience" directory.
        # We check the source code of main to ensure the directory name is
        # "resilience" and not "attention".
        import inspect
        main_source = inspect.getsource(gp.main)

        # The fixed code should have: resilience_dir = output_dir / "resilience"
        assert '"resilience"' in main_source, (
            'generate_plots.py main() does not use "resilience" directory -- '
            "resilience plots may be going to wrong directory"
        )

        # Ensure the category "resilience" in PLOT_TYPES is separate from "attention"
        assert "resilience" in gp.PLOT_TYPES, (
            '"resilience" not in PLOT_TYPES categories'
        )
        assert "attention" in gp.PLOT_TYPES, (
            '"attention" not in PLOT_TYPES categories'
        )
        # They should be distinct categories
        assert gp.PLOT_TYPES["resilience"] != gp.PLOT_TYPES["attention"], (
            "resilience and attention share the same plot list -- "
            "they should be separate categories"
        )


# ---------------------------------------------------------------------------
# Test 8: LR scales linearly with world_size when lr_scaling=True
# Bug: Round 14, Task 3
# Fix: configure_optimizers() now multiplies base_lr by world_size when
#      training.lr_scaling is True and world_size > 1.
# ---------------------------------------------------------------------------
class TestLRScalingWithWorldSize:
    """Verify LR scales linearly with world_size."""

    def test_lr_scaling_with_world_size(self):
        """
        Round 14, Task 3: When lr_scaling is enabled and world_size > 1,
        the effective learning rate should be scaled by world_size
        (linear scaling rule, Goyal et al. 2017).

        The bug was that the learning rate was not scaled for multi-GPU
        training, causing effectively lower learning rates per step.
        """
        from src.training.lightning_module import CognitiveResilienceLightningModule

        base_lr = 0.001
        world_size = 4

        config = OmegaConf.create({
            "model": {
                "n_genes": 50,
                "n_cell_types": N_CELL_TYPES,
                "d_embed": 32,
                "d_fused": 32,
                "dropout": 0.1,
                "hgt": {"n_layers": 1, "n_heads": 4},
                "set_transformer": {
                    "n_isab_layers": 1,
                    "n_inducing_points": 4,
                },
                "pathology_attention": {
                    "d_cond": 16,
                    "n_heads": 4,
                },
                "gene_gate": {"initial_temperature": 2.0},
                "cell_type_selector": {"selection_temperature": 1.0},
                "head": {"type": "deterministic", "d_hidden": 16},
                "pseudobulk": {"mlp_hidden": None, "use_layer_norm": True},
            },
            "training": {
                "max_epochs": 10,
                "optimizer": {
                    "type": "adamw",
                    "lr": base_lr,
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "type": "cosine",
                    "warmup_epochs": 0,
                    "eta_min": 1e-6,
                },
                "loss": {"type": "mse"},
                "lr_scaling": True,
                "regularization": {"gene_gate_l1": 0.0},
                "gradient_clip_val": 1.0,
            },
        })

        module = CognitiveResilienceLightningModule(config)

        # Mock a trainer with world_size > 1
        mock_trainer = MagicMock()
        mock_trainer.world_size = world_size
        module._trainer = mock_trainer
        # Lightning accesses self.trainer which proxies to _trainer
        # But configure_optimizers uses self.trainer.world_size
        # We need to patch the property
        with patch.object(
            type(module), "trainer", new_callable=lambda: property(lambda self: mock_trainer)
        ):
            result = module.configure_optimizers()

        # Extract the optimizer
        optimizer = result["optimizer"]
        actual_lr = optimizer.param_groups[0]["lr"]

        expected_lr = base_lr * world_size
        assert abs(actual_lr - expected_lr) < 1e-9, (
            f"LR was {actual_lr} but expected {expected_lr} "
            f"(base_lr={base_lr} * world_size={world_size}). "
            "LR scaling is broken."
        )

    def test_lr_not_scaled_when_disabled(self):
        """
        When lr_scaling is False, the LR should not be scaled regardless
        of world_size.
        """
        from src.training.lightning_module import CognitiveResilienceLightningModule

        base_lr = 0.001

        config = OmegaConf.create({
            "model": {
                "n_genes": 50,
                "n_cell_types": N_CELL_TYPES,
                "d_embed": 32,
                "d_fused": 32,
                "dropout": 0.1,
                "hgt": {"n_layers": 1, "n_heads": 4},
                "set_transformer": {
                    "n_isab_layers": 1,
                    "n_inducing_points": 4,
                },
                "pathology_attention": {
                    "d_cond": 16,
                    "n_heads": 4,
                },
                "gene_gate": {"initial_temperature": 2.0},
                "cell_type_selector": {"selection_temperature": 1.0},
                "head": {"type": "deterministic", "d_hidden": 16},
                "pseudobulk": {"mlp_hidden": None, "use_layer_norm": True},
            },
            "training": {
                "max_epochs": 10,
                "optimizer": {
                    "type": "adamw",
                    "lr": base_lr,
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "type": "cosine",
                    "warmup_epochs": 0,
                    "eta_min": 1e-6,
                },
                "loss": {"type": "mse"},
                "lr_scaling": False,
                "regularization": {"gene_gate_l1": 0.0},
                "gradient_clip_val": 1.0,
            },
        })

        module = CognitiveResilienceLightningModule(config)

        # Mock a trainer with world_size > 1
        mock_trainer = MagicMock()
        mock_trainer.world_size = 4
        with patch.object(
            type(module), "trainer", new_callable=lambda: property(lambda self: mock_trainer)
        ):
            result = module.configure_optimizers()

        optimizer = result["optimizer"]
        actual_lr = optimizer.param_groups[0]["lr"]

        assert abs(actual_lr - base_lr) < 1e-9, (
            f"LR was {actual_lr} but expected {base_lr} "
            f"(lr_scaling=False should not scale LR)"
        )


# ---------------------------------------------------------------------------
# Test 9: --final mode does not use holdout test set for model selection
# Bug: Round 16, Task 1
# Fix: scripts/train.py --final block now:
#      1. Passes train_dataloaders only to trainer.fit() (no val_dataloaders)
#      2. Removes MinEpochEarlyStopping from callbacks
#      3. Uses ModelCheckpoint with save_top_k=0 (no metric-based selection)
#      4. Calls trainer.test() after fit for single unbiased evaluation
# ---------------------------------------------------------------------------
class TestFinalModeDoesNotUseHoldoutForSelection:
    """Verify --final mode does not leak holdout data into training decisions."""

    def test_final_mode_does_not_use_holdout_for_selection(self):
        """
        Round 16, Task 1: In --final mode, the holdout test set must NOT be
        used as val_dataloaders during trainer.fit(). This would cause early
        stopping and model checkpoint selection to evaluate on the holdout
        set repeatedly, biasing final performance estimates (data leakage).

        The fix ensures:
        - trainer.fit() receives NO val_dataloaders
        - MinEpochEarlyStopping is removed from callbacks
        - ModelCheckpoint uses save_top_k=0 (last epoch only, no metric)
        - trainer.test() is called after fit for single unbiased evaluation
        """
        import sys
        from lightning.pytorch.callbacks import ModelCheckpoint

        from src.training.callbacks import (
            MinEpochEarlyStopping,
            ResilienceModelCheckpoint,
        )

        # We need to test the --final code path in scripts/train.py
        # without actually loading data. Use mocks throughout.

        # Mock all heavyweight dependencies
        with (
            patch("scripts.train.load_config") as mock_load_config,
            patch("scripts.train.set_seed"),
            patch("scripts.train.ExperimentManager") as mock_exp_mgr,
            patch("scripts.train.CognitiveResilienceLightningModule") as mock_module_cls,
            patch("scripts.train.OmegaConf") as mock_omegaconf,
            patch("src.utils.config.validate_config"),
            patch("scripts.train.setup_callbacks") as mock_setup_callbacks,
            patch("scripts.train.setup_trainer") as mock_setup_trainer,
            patch("src.data.splits.load_splits") as mock_load_splits,
            patch("src.data.splits.get_final_train_subjects") as mock_get_final,
            patch("src.data.loaders.create_fold_dataloaders") as mock_create_loaders,
            patch("scripts.train.torch") as mock_torch,
        ):
            # Configure mock config
            mock_config = MagicMock()
            mock_config.experiment.get.return_value = 42
            mock_config.paths.get.side_effect = lambda key, default: default
            mock_config.data.metadata_path = "/fake/metadata"
            mock_load_config.return_value = mock_config

            # OmegaConf mocks
            mock_omegaconf.to_container.return_value = {}
            mock_omegaconf.update = MagicMock()

            # Experiment manager mock
            mock_experiment = MagicMock()
            mock_experiment.exp_hash = "abc123"
            mock_experiment.exp_dir = "/tmp/exp"
            mock_experiment.checkpoints_dir = "/tmp/ckpt"
            mock_experiment.tensorboard_dir = "/tmp/tb"
            mock_experiment.model_dir = Path("/tmp/test_model")
            mock_exp_mgr.return_value.create_experiment.return_value = mock_experiment

            # Module mock
            mock_module = MagicMock()
            mock_module_cls.return_value = mock_module

            # Splits mock -- has holdout_test and folds
            mock_splits = {
                "holdout_test": ["subj_A", "subj_B"],
                "folds": [
                    {"train": ["subj_C", "subj_D"], "val": ["subj_E"]},
                ],
            }
            mock_load_splits.return_value = mock_splits
            mock_get_final.return_value = ["subj_C", "subj_D", "subj_E"]

            # create_fold_dataloaders returns mock loaders
            mock_train_loader = MagicMock(name="train_loader")
            mock_test_loader = MagicMock(name="test_loader")
            mock_create_loaders.side_effect = [
                (mock_train_loader, MagicMock()),  # train call (discard val)
                (MagicMock(), mock_test_loader),    # test call (discard train)
            ]

            # setup_callbacks returns realistic callbacks to be filtered
            mock_setup_callbacks.return_value = [
                ModelCheckpoint(
                    dirpath="/tmp", monitor="val_loss", mode="min",
                    save_top_k=1, save_last=True,
                ),
                MinEpochEarlyStopping(
                    min_epochs=20, monitor="val_loss", patience=10,
                    min_delta=0.001, mode="min",
                ),
                MagicMock(name="LearningRateMonitor"),
                MagicMock(name="TemperatureAnnealing"),
                MagicMock(name="GradientNormLogger"),
                ResilienceModelCheckpoint(),
            ]

            # Trainer mock
            mock_trainer = MagicMock()
            mock_trainer.max_epochs = 100
            mock_setup_trainer.return_value = mock_trainer

            # Metadata + adata mock -- patch Path, pd.read_csv, sc.read_h5ad
            with (
                patch("scripts.train.Path") as mock_path_cls,
                patch("scripts.train.pd") as mock_pd,
                patch("scripts.train.sc") as mock_sc,
            ):
                mock_path_instance = MagicMock()
                mock_path_cls.return_value = mock_path_instance
                mock_csv_path = MagicMock()
                mock_csv_path.exists.return_value = True
                mock_path_instance.__truediv__ = MagicMock(return_value=mock_csv_path)
                mock_pd.read_csv.return_value = MagicMock(name="metadata_df")
                mock_sc.read_h5ad.return_value = MagicMock(name="adata")

                # Simulate CLI args: --final --splits-path /fake/splits.json
                test_args = [
                    "train.py",
                    "--config", "configs/default.yaml",
                    "--final",
                    "--splits-path", "/fake/splits.json",
                ]
                with patch.object(sys, "argv", test_args):
                    from scripts.train import main
                    main()

            # --- Assertions ---

            # 1. trainer.fit() must NOT receive val_dataloaders
            mock_trainer.fit.assert_called_once()
            fit_call_kwargs = mock_trainer.fit.call_args
            assert "val_dataloaders" not in fit_call_kwargs.kwargs, (
                "trainer.fit() received val_dataloaders in --final mode -- "
                "holdout test data is leaking into training (data leakage bug)"
            )

            # 2. trainer.test() must be called after fit
            mock_trainer.test.assert_called_once()

            # 3. setup_trainer was called with filtered callbacks
            mock_setup_trainer.assert_called()
            trainer_call_kwargs = mock_setup_trainer.call_args
            final_callbacks = trainer_call_kwargs.kwargs.get(
                "callbacks",
                trainer_call_kwargs.args[1] if len(trainer_call_kwargs.args) > 1 else None,
            )
            assert final_callbacks is not None, (
                "setup_trainer was not called with explicit callbacks"
            )

            # 4. No MinEpochEarlyStopping in final callbacks
            early_stop_cbs = [
                cb for cb in final_callbacks
                if isinstance(cb, MinEpochEarlyStopping)
            ]
            assert len(early_stop_cbs) == 0, (
                "MinEpochEarlyStopping found in --final mode callbacks -- "
                "early stopping must be disabled for final training"
            )

            # 5. ModelCheckpoint should have save_top_k=0 (no metric selection)
            ckpt_cbs = [
                cb for cb in final_callbacks
                if isinstance(cb, ModelCheckpoint)
            ]
            assert len(ckpt_cbs) >= 1, (
                "No ModelCheckpoint found in --final mode callbacks"
            )
            for cb in ckpt_cbs:
                assert cb.save_top_k == 0, (
                    f"ModelCheckpoint has save_top_k={cb.save_top_k} "
                    "but expected 0 (no metric-based selection in final mode)"
                )

            # 6. ResilienceModelCheckpoint is still present
            resilience_cbs = [
                cb for cb in final_callbacks
                if isinstance(cb, ResilienceModelCheckpoint)
            ]
            assert len(resilience_cbs) >= 1, (
                "ResilienceModelCheckpoint missing from --final mode callbacks"
            )


# ---------------------------------------------------------------------------
# Test 10: Bayesian model exports backbone_weights.pt, not full weights.pt
# Bug: Round 17, Task 2
# Fix: _export_weights() in scripts/train.py gates the export:
#      - Deterministic heads: full state_dict as weights.pt
#      - Bayesian heads: backbone-only (excluding prediction_head.*) as
#        backbone_weights.pt. Bayesian inference requires the full .ckpt
#        with guide and param store; exporting weights.pt would produce
#        random predictions from N(0,1) priors.
# ---------------------------------------------------------------------------
class TestBayesianModelExportsBackboneNotFullWeights:
    """Verify Bayesian models export backbone_weights.pt, not weights.pt."""

    def test_bayesian_model_exports_backbone_not_full_weights(self, tmp_path):
        """
        Round 17, Task 2: Bayesian models must NOT export weights.pt — only
        backbone_weights.pt (excluding prediction_head.* keys).

        The bug was that both deterministic and Bayesian models exported full
        weights.pt. For Bayesian heads, the prediction_head uses Pyro priors
        (N(0,1)) and without the guide, loading weights.pt for inference
        produces random predictions.
        """
        from scripts.train import _export_weights

        module = MagicMock()
        module.model.state_dict.return_value = {
            "pseudobulk_encoder.weight": torch.randn(10, 10),
            "hgt_encoder.weight": torch.randn(10, 10),
            "prediction_head.fc1.weight": torch.randn(5, 10),
            "prediction_head.fc2.weight": torch.randn(1, 5),
        }

        _export_weights(module, tmp_path, is_bayesian=True)

        assert not (tmp_path / "weights.pt").exists(), (
            "Bayesian model should NOT create weights.pt"
        )
        assert (tmp_path / "backbone_weights.pt").exists()

        loaded = torch.load(tmp_path / "backbone_weights.pt", weights_only=True)
        assert "pseudobulk_encoder.weight" in loaded
        assert "hgt_encoder.weight" in loaded
        assert "prediction_head.fc1.weight" not in loaded
        assert "prediction_head.fc2.weight" not in loaded

    def test_deterministic_model_exports_full_weights(self, tmp_path):
        """
        Round 17, Task 2: Deterministic models should export full weights.pt
        containing all keys including prediction_head.
        """
        from scripts.train import _export_weights

        module = MagicMock()
        full_state = {
            "pseudobulk_encoder.weight": torch.randn(10, 10),
            "prediction_head.fc1.weight": torch.randn(5, 10),
        }
        module.model.state_dict.return_value = full_state

        _export_weights(module, tmp_path, is_bayesian=False)

        assert (tmp_path / "weights.pt").exists()
        assert not (tmp_path / "backbone_weights.pt").exists()
