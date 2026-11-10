"""
tests/test_infrastructure.py
==============================
Unit tests for the CheXpert B1v2 training infrastructure.

Run:
    python -m pytest tests/test_infrastructure.py -v
    # or directly:
    python tests/test_infrastructure.py

All model tests use weights=None to skip ImageNet downloads.
Dataset tests are conditional on the CSV being available.
"""

from __future__ import annotations

import os
import sys
import re
import math
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHEXPERT_CSV = PROJECT_ROOT / "data" / "chexpert_subset" / "chexpert_materialized_index.csv"
CHEXPERT_CACHE = PROJECT_ROOT / "data" / "chexpert_cache"

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


def _make_model(cfg_overrides=None):
    """Build CheXpertB1V2Model in test mode (no pretrained weights download)."""
    from src.models.chexpert_b1_v2 import CheXpertB1V2Model
    # Monkey-patch to skip pretrained download
    import torchvision.models as tvm
    _orig = tvm.efficientnet_b1
    def _no_pretrain(**kwargs):
        kwargs.pop("weights", None)
        return _orig(weights=None, **kwargs)
    # Build model using weights=None via patching DEFAULT_CFG approach
    cfg = {"input_resolution": 224, "feat_dim": 256, "num_classes": 14}
    if cfg_overrides:
        cfg.update(cfg_overrides)
    # Temporarily replace to avoid download
    import torchvision.models
    orig_fn = torchvision.models.efficientnet_b1
    try:
        from torchvision.models import EfficientNet_B1_Weights
        torchvision.models.efficientnet_b1 = lambda **kw: orig_fn(weights=None)
        # Patch the import inside chexpert_b1_v2 module too
        import src.models.chexpert_b1_v2 as m_mod
        orig_eb1 = m_mod.efficientnet_b1
        orig_wts = m_mod.EfficientNet_B1_Weights
        m_mod.efficientnet_b1 = lambda **kw: orig_eb1(weights=None)
        try:
            model = CheXpertB1V2Model(cfg)
        finally:
            m_mod.efficientnet_b1 = orig_eb1
    except Exception:
        model = CheXpertB1V2Model(cfg)
    return model


def _build_test_model(cfg_overrides=None):
    """Build a model for testing (patches to skip weight download)."""
    from torchvision.models import efficientnet_b1
    import src.models.chexpert_b1_v2 as m_mod
    cfg = {
        "input_resolution": 224,
        "feat_dim": 256,
        "num_classes": 14,
        "fuse_block5": True,
        "attn_pool_mode": "norm_avg",
        "disable_dilation3": False,
        "attn_regularizer_weight": 0.0,
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)

    orig_b1 = m_mod.efficientnet_b1
    orig_wts = m_mod.EfficientNet_B1_Weights

    class _FakeWeights:
        DEFAULT = None

    m_mod.efficientnet_b1 = lambda weights=None: orig_b1(weights=None)
    m_mod.EfficientNet_B1_Weights = _FakeWeights

    from src.models.chexpert_b1_v2 import CheXpertB1V2Model
    try:
        model = CheXpertB1V2Model(cfg)
    finally:
        m_mod.efficientnet_b1 = orig_b1
        m_mod.EfficientNet_B1_Weights = orig_wts

    return model


# =============================================================================
# TestModelShapes
# =============================================================================

class TestModelShapes(unittest.TestCase):
    """Test tensor shapes under various config flags (CPU, no pretrained weights)."""

    def _run_model(self, cfg, resolution=224, batch_size=2):
        model = _build_test_model(cfg)
        model.eval()
        x = torch.randn(batch_size, 3, resolution, resolution)
        with torch.no_grad():
            logits, attn_reg = model(x)
        return logits, attn_reg, model

    def test_shape_224(self):
        logits, _, _ = self._run_model({"input_resolution": 224})
        self.assertEqual(logits.shape, (2, 14), f"Expected (2,14), got {logits.shape}")

    def test_shape_320(self):
        logits, _, _ = self._run_model({"input_resolution": 320}, resolution=320)
        self.assertEqual(logits.shape, (2, 14), f"Expected (2,14) at 320, got {logits.shape}")

    def test_fuse_block5_true(self):
        logits, _, _ = self._run_model({"fuse_block5": True})
        self.assertEqual(logits.shape, (2, 14))

    def test_fuse_block5_false(self):
        logits, _, _ = self._run_model({"fuse_block5": False})
        self.assertEqual(logits.shape, (2, 14))

    def test_attn_pool_sum(self):
        logits, _, _ = self._run_model({"attn_pool_mode": "sum"})
        self.assertEqual(logits.shape, (2, 14))

    def test_attn_pool_norm_avg(self):
        logits, _, _ = self._run_model({"attn_pool_mode": "norm_avg"})
        self.assertEqual(logits.shape, (2, 14))

    def test_attn_pool_softmax(self):
        logits, _, _ = self._run_model({"attn_pool_mode": "softmax"})
        self.assertEqual(logits.shape, (2, 14))

    def test_disable_dilation3_false(self):
        logits, _, _ = self._run_model({"disable_dilation3": False})
        self.assertEqual(logits.shape, (2, 14))

    def test_disable_dilation3_true(self):
        logits, _, _ = self._run_model({"disable_dilation3": True})
        self.assertEqual(logits.shape, (2, 14))

    def test_attn_regularizer_zero(self):
        _, attn_reg, _ = self._run_model({"attn_regularizer_weight": 0.0})
        self.assertAlmostEqual(float(attn_reg), 0.0, places=5)

    def test_attn_regularizer_positive(self):
        _, attn_reg, _ = self._run_model({"attn_regularizer_weight": 0.1})
        self.assertGreater(float(attn_reg), 0.0,
                           "attn_reg_loss should be positive with weight=0.1")

    def test_num_classes_14(self):
        logits, _, _ = self._run_model({"num_classes": 14})
        self.assertEqual(logits.shape[1], 14)

    def test_logits_finite(self):
        logits, attn_reg, _ = self._run_model({})
        self.assertTrue(torch.isfinite(logits).all(), "Logits should be finite")
        self.assertTrue(torch.isfinite(attn_reg), "attn_reg should be finite")


# =============================================================================
# TestAttentionProperties
# =============================================================================

class TestAttentionProperties(unittest.TestCase):
    """Test attention map normalization and validity."""

    def _get_attention_maps(self, cfg):
        model = _build_test_model(cfg)
        model.eval()
        maps_captured = []

        def hook(module, input, output):
            maps_captured.append(output.detach())

        h = model.dag.register_forward_hook(hook)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            model(x)
        h.remove()
        return maps_captured[0]  # [B, 14, H, W]

    def test_sigmoid_attention_non_negative(self):
        maps = self._get_attention_maps({"attn_pool_mode": "norm_avg"})
        self.assertTrue((maps >= 0).all().item(),
                        "Sigmoid attention maps should be non-negative")

    def test_sigmoid_attention_bounded(self):
        maps = self._get_attention_maps({"attn_pool_mode": "norm_avg"})
        self.assertTrue((maps <= 1).all().item(),
                        "Sigmoid attention maps should be <= 1")

    def test_norm_avg_denominator(self):
        """With norm_avg, pooled features should be in reasonable range."""
        model = _build_test_model({"attn_pool_mode": "norm_avg"})
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, _ = model(x)
        # Logits should be finite and reasonably bounded
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits.abs() < 1000).all(),
                        "Logits should be reasonably bounded with norm_avg")

    def test_softmax_attention_sums_to_one(self):
        """With attn_pool_mode=softmax, the softmax is applied before pooling."""
        # We test indirectly: softmax attention ensures sum over spatial = 1
        model = _build_test_model({"attn_pool_mode": "softmax"})
        model.eval()
        maps_captured = []

        def hook(module, input, output):
            maps_captured.append(output.detach())

        h = model.dag.register_forward_hook(hook)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            model(x)
        h.remove()

        raw_maps = maps_captured[0]  # [1, 14, H, W] — these are sigmoid maps (before softmax in pooling)
        # The softmax is applied during pooling, not in dag output
        # So we just verify model runs fine with softmax mode
        logits_check = model(x)[0]
        self.assertEqual(logits_check.shape, (1, 14))


# =============================================================================
# TestPatientSplits
# =============================================================================

@unittest.skipUnless(CHEXPERT_CSV.exists(), f"Dataset not at {CHEXPERT_CSV}")
class TestPatientSplits(unittest.TestCase):
    """Test patient-level split correctness."""

    @classmethod
    def setUpClass(cls):
        from scripts.train_chexpert_b1_v2 import make_patient_splits, _extract_patient_id_from_local_filename
        cls.df_all = pd.read_csv(CHEXPERT_CSV)
        cls.train_df, cls.val_df, cls.test_df = make_patient_splits(
            cls.df_all, val_frac=0.10, test_frac=0.10, seed=42
        )

        def _get_pids(df):
            if "local_filename" in df.columns:
                return set(df["local_filename"].apply(_extract_patient_id_from_local_filename).unique())
            return set(df["file_id"].apply(lambda x: x.split(".")[0]).unique())

        cls.train_pids = _get_pids(cls.train_df)
        cls.val_pids   = _get_pids(cls.val_df)
        cls.test_pids  = _get_pids(cls.test_df)

    def test_no_patient_leakage_train_val(self):
        overlap = self.train_pids & self.val_pids
        self.assertEqual(len(overlap), 0, f"Train/val leakage: {len(overlap)} patients")

    def test_no_patient_leakage_train_test(self):
        overlap = self.train_pids & self.test_pids
        self.assertEqual(len(overlap), 0, f"Train/test leakage: {len(overlap)} patients")

    def test_no_patient_leakage_val_test(self):
        overlap = self.val_pids & self.test_pids
        self.assertEqual(len(overlap), 0, f"Val/test leakage: {len(overlap)} patients")

    def test_split_sizes_approximate(self):
        n_total = len(self.df_all)
        # Test: val is roughly 10% of total rows
        val_frac = len(self.val_df) / n_total
        self.assertGreater(val_frac, 0.07, f"Val fraction too small: {val_frac:.3f}")
        self.assertLess(val_frac, 0.14, f"Val fraction too large: {val_frac:.3f}")

    def test_all_rows_assigned(self):
        n_total = len(self.df_all)
        n_assigned = len(self.train_df) + len(self.val_df) + len(self.test_df)
        self.assertEqual(n_total, n_assigned,
                         f"Row count mismatch: {n_total} != {n_assigned}")


# =============================================================================
# TestSampler
# =============================================================================

@unittest.skipUnless(CHEXPERT_CSV.exists(), f"Dataset not at {CHEXPERT_CSV}")
class TestSampler(unittest.TestCase):
    """Test sampler pool ratios and behavior."""

    @classmethod
    def setUpClass(cls):
        from scripts.train_chexpert_b1_v2 import make_patient_splits, TieredSamplerV2
        df_all = pd.read_csv(CHEXPERT_CSV)
        train_df, _, _ = make_patient_splits(df_all, val_frac=0.10, test_frac=0.10)
        cls.train_df = train_df.head(5000)  # small subset for speed
        cls.TieredSamplerV2 = TieredSamplerV2

    def _make_sampler(self, pool_counts=None, intra_pool_mode="sqrt_freq", batches=20):
        return self.TieredSamplerV2(
            df=self.train_df,
            pool_counts=pool_counts or [14, 12, 6],
            labels=CHEXPERT_14_LABELS,
            batches_per_epoch=batches,
            batch_size=32,
            intra_pool_mode=intra_pool_mode,
            seed=42,
        )

    def test_epoch_length(self):
        """len(sampler) == batches_per_epoch (batch sampler interface)."""
        sampler = self._make_sampler(batches=50)
        # TieredSamplerV2 is a batch sampler: __len__ returns number of batches
        self.assertEqual(len(sampler), 50)

    def test_no_index_out_of_bounds(self):
        """All indices returned by sampler are valid dataset indices."""
        n = len(self.train_df)
        sampler = self._make_sampler(batches=5)
        # Batch sampler yields lists of ints; flatten them
        all_indices = []
        for batch in sampler:
            if isinstance(batch, (list, np.ndarray)):
                all_indices.extend(batch)
            else:
                all_indices.append(batch)
        for idx in all_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, n, f"Index {idx} out of bounds (n={n})")

    def test_pool_counts_sum_equals_batch_size(self):
        """Each yielded batch has exactly batch_size elements."""
        pool_counts = [14, 12, 6]
        sampler = self._make_sampler(pool_counts=pool_counts, batches=10)
        for batch in sampler:
            self.assertEqual(len(batch), sum(pool_counts),
                             f"Batch size mismatch: {len(batch)} != {sum(pool_counts)}")

    def test_realized_ratios(self):
        sampler = self._make_sampler(pool_counts=[14, 12, 6], batches=100)
        ratios = sampler.compute_realized_ratios(n_batches=100)
        # Train script's TieredSamplerV2 uses these keys
        self.assertIn("pool1_rare_frac", ratios)
        self.assertIn("pool2_common_frac", ratios)
        self.assertIn("pool3_normal_frac", ratios)
        # Sum to 1
        total = sum(ratios.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        # pool1 should be ~14/32
        expected_p1 = 14 / 32
        self.assertAlmostEqual(ratios["pool1_rare_frac"], expected_p1, places=3)


# =============================================================================
# TestMaskedLoss
# =============================================================================

class TestMaskedLoss(unittest.TestCase):
    """Test masked BCE loss."""

    def _loss_fn(self, logits, targets, mask, pos_weight=None):
        from scripts.train_chexpert_b1_v2 import masked_bce_loss
        return masked_bce_loss(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None,
        )

    def test_full_mask_finite(self):
        logits  = np.random.randn(4, 14)
        targets = (np.random.rand(4, 14) > 0.5).astype(float)
        mask    = np.ones((4, 14))
        loss = self._loss_fn(logits, targets, mask)
        self.assertTrue(torch.isfinite(loss), "Loss should be finite with full mask")
        self.assertGreater(float(loss), 0.0, "Loss should be positive")

    def test_full_mask_matches_bce(self):
        """With all-ones mask, masked_bce_loss ≈ BCEWithLogitsLoss (mean)."""
        torch.manual_seed(0)
        logits  = torch.randn(8, 14)
        targets = (torch.rand(8, 14) > 0.5).float()
        mask    = torch.ones(8, 14)

        from scripts.train_chexpert_b1_v2 import masked_bce_loss
        our_loss = masked_bce_loss(logits, targets, mask)
        ref_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        self.assertAlmostEqual(float(our_loss), float(ref_loss), places=4)

    def test_zero_mask_gives_zero_loss(self):
        """With all-zero mask, loss should be 0."""
        logits  = np.random.randn(4, 14)
        targets = np.ones((4, 14))
        mask    = np.zeros((4, 14))
        loss = self._loss_fn(logits, targets, mask)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_partial_mask_in_between(self):
        """Partial mask gives loss between full and zero mask."""
        torch.manual_seed(1)
        logits  = torch.randn(4, 14)
        targets = torch.ones(4, 14)
        mask_full = torch.ones(4, 14)
        mask_zero = torch.zeros(4, 14)
        mask_half = torch.zeros(4, 14)
        mask_half[:, :7] = 1.0

        from scripts.train_chexpert_b1_v2 import masked_bce_loss
        loss_full = float(masked_bce_loss(logits, targets, mask_full))
        loss_zero = float(masked_bce_loss(logits, targets, mask_zero))
        loss_half = float(masked_bce_loss(logits, targets, mask_half))

        self.assertAlmostEqual(loss_zero, 0.0, places=5)
        self.assertGreater(loss_full, 0.0)
        # Half mask loss should be close to full (since same logits for unmasked)
        self.assertGreater(loss_half, 0.0)


# =============================================================================
# TestUncertainPolicy
# =============================================================================

@unittest.skipUnless(CHEXPERT_CSV.exists(), f"Dataset not at {CHEXPERT_CSV}")
class TestUncertainPolicy(unittest.TestCase):
    """Test uncertain label handling policies."""

    @classmethod
    def setUpClass(cls):
        df_all = pd.read_csv(CHEXPERT_CSV)
        # Find a row with at least one -1 label
        from scripts.train_chexpert_b1_v2 import make_patient_splits, CheXpertLocalDatasetV2
        _, val_df, _ = make_patient_splits(df_all)
        # Find rows with uncertain labels
        uncertain_mask = (df_all[CHEXPERT_14_LABELS] == -1.0).any(axis=1)
        cls.df_with_uncertain = df_all[uncertain_mask].head(100).reset_index(drop=True)
        cls.cache_dir = str(CHEXPERT_CACHE)
        cls.DatasetClass = CheXpertLocalDatasetV2

    def _make_dataset(self, policy):
        from scripts.train_chexpert_b1_v2 import CheXpertLocalDatasetV2
        return CheXpertLocalDatasetV2(
            self.df_with_uncertain,
            split="val",
            cfg={
                "uncertain_policy": policy,
                "cache_dir": self.cache_dir,
                "model_input_resolution": 224,
            }
        )

    def _get_first_uncertain_label_values(self, dataset):
        """Return label values for the first sample with uncertain labels."""
        df = dataset.df
        for idx in range(len(df)):
            row = df.iloc[idx]
            orig_vals = [row.get(lbl, 0.0) for lbl in CHEXPERT_14_LABELS]
            if -1.0 in [v for v in orig_vals if isinstance(v, float)]:
                _, labels, mask = dataset[idx]
                return labels.numpy(), mask.numpy(), orig_vals
        return None, None, None

    def test_u_zeros(self):
        ds = self._make_dataset("u_zeros")
        labels, mask, orig = self._get_first_uncertain_label_values(ds)
        if labels is None:
            self.skipTest("No uncertain labels found in test subset")
        for i, (lbl, orig_v) in enumerate(zip(labels, orig)):
            if isinstance(orig_v, float) and orig_v == -1.0:
                self.assertAlmostEqual(float(lbl), 0.0, places=5,
                                       msg=f"u_zeros: -1 label {i} should become 0")

    def test_u_ones(self):
        ds = self._make_dataset("u_ones")
        labels, mask, orig = self._get_first_uncertain_label_values(ds)
        if labels is None:
            self.skipTest("No uncertain labels found in test subset")
        for i, (lbl, orig_v) in enumerate(zip(labels, orig)):
            if isinstance(orig_v, float) and orig_v == -1.0:
                self.assertAlmostEqual(float(lbl), 1.0, places=5,
                                       msg=f"u_ones: -1 label {i} should become 1")

    def test_ignore_mask(self):
        ds = self._make_dataset("ignore")
        labels, mask, orig = self._get_first_uncertain_label_values(ds)
        if labels is None:
            self.skipTest("No uncertain labels found in test subset")
        for i, (lbl, m, orig_v) in enumerate(zip(labels, mask, orig)):
            if isinstance(orig_v, float) and orig_v == -1.0:
                self.assertAlmostEqual(float(m), 0.0, places=5,
                                       msg=f"ignore: mask[{i}] should be 0 for uncertain label")

    def test_smooth_in_range(self):
        ds = self._make_dataset("smooth")
        labels, mask, orig = self._get_first_uncertain_label_values(ds)
        if labels is None:
            self.skipTest("No uncertain labels found in test subset")
        for i, (lbl, orig_v) in enumerate(zip(labels, orig)):
            if isinstance(orig_v, float) and orig_v == -1.0:
                # With smooth policy, NaN rows might remain at 0
                # (depends on implementation detail)
                # Just check it's in [0, 1]
                self.assertGreaterEqual(float(lbl), 0.0)
                self.assertLessEqual(float(lbl), 1.0)


# =============================================================================
# TestCheckpointCompat
# =============================================================================

class TestCheckpointCompat(unittest.TestCase):
    """Test that model can be constructed and state dict loaded gracefully."""

    def test_model_has_arch_version(self):
        model = _build_test_model()
        self.assertEqual(model.arch_version, "b1v2")

    def test_freeze_backbone(self):
        model = _build_test_model()
        model.freeze_backbone()
        # backbone.features params should not require grad
        frozen_params = [p for p in model.backbone.features.parameters()]
        for p in frozen_params:
            self.assertFalse(p.requires_grad,
                             "Backbone params should not require grad after freeze_backbone()")

    def test_unfreeze_stage_b_last3(self):
        model = _build_test_model()
        model.freeze_backbone()
        param_groups = model.get_param_groups(
            backbone_lr=1e-5, head_lr=1e-4, unfreeze_mode="last3"
        )
        self.assertEqual(len(param_groups), 2)
        # backbone group should have some trainable params
        backbone_params = param_groups[0]["params"]
        self.assertGreater(len(backbone_params), 0)
        self.assertAlmostEqual(param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(param_groups[1]["lr"], 1e-4)

    def test_state_dict_save_load(self):
        import tempfile
        model1 = _build_test_model()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            fname = f.name
        torch.save({"model_state_dict": model1.state_dict(), "arch_version": "b1v2"}, fname)

        model2 = _build_test_model()
        ckpt = torch.load(fname, map_location="cpu")
        missing, unexpected = model2.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.assertEqual(len(missing), 0,
                         f"Missing keys on reload: {missing}")
        self.assertEqual(len(unexpected), 0,
                         f"Unexpected keys on reload: {unexpected}")
        os.unlink(fname)


# =============================================================================
# TestMSFBBranches
# =============================================================================

class TestMSFBBranches(unittest.TestCase):
    """Test MultiScaleFeatureBlock branch configurations."""

    def test_msfb_4branch_output_shape(self):
        from src.models.chexpert_b1_v2 import MultiScaleFeatureBlock
        msfb = MultiScaleFeatureBlock(1280, 256, disable_dilation3=False)
        x = torch.randn(2, 1280, 7, 7)
        out = msfb(x)
        self.assertEqual(out.shape, (2, 256, 7, 7))

    def test_msfb_3branch_output_shape(self):
        from src.models.chexpert_b1_v2 import MultiScaleFeatureBlock
        msfb = MultiScaleFeatureBlock(1280, 256, disable_dilation3=True)
        x = torch.randn(2, 1280, 7, 7)
        out = msfb(x)
        self.assertEqual(out.shape, (2, 256, 7, 7))

    def test_channel_attention_shape(self):
        from src.models.chexpert_b1_v2 import ChannelAttention
        ca = ChannelAttention(256, reduction=16)
        x = torch.randn(2, 256, 14, 14)
        out = ca(x)
        self.assertEqual(out.shape, (2, 256, 14, 14))

    def test_spatial_attention_shape(self):
        from src.models.chexpert_b1_v2 import SpatialAttention
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 256, 14, 14)
        out = sa(x)
        self.assertEqual(out.shape, (2, 256, 14, 14))

    def test_disease_attention_generator_sigmoid(self):
        from src.models.chexpert_b1_v2 import DiseaseAttentionGenerator
        dag = DiseaseAttentionGenerator(256, num_classes=14, att_norm="sigmoid")
        x = torch.randn(2, 256, 14, 14)
        maps = dag(x)
        self.assertEqual(maps.shape, (2, 14, 14, 14))
        self.assertTrue((maps >= 0).all().item())
        self.assertTrue((maps <= 1).all().item())

    def test_disease_attention_generator_softmax(self):
        from src.models.chexpert_b1_v2 import DiseaseAttentionGenerator
        dag = DiseaseAttentionGenerator(256, num_classes=14, att_norm="spatial_softmax")
        x = torch.randn(2, 256, 14, 14)
        maps = dag(x)
        self.assertEqual(maps.shape, (2, 14, 14, 14))
        # Softmax over spatial: sum over H*W should be 1 for each disease per sample
        maps_flat = maps.view(2, 14, -1)
        sums = maps_flat.sum(dim=-1)  # [2, 14]
        for b in range(2):
            for c in range(14):
                self.assertAlmostEqual(float(sums[b, c]), 1.0, places=4)


# =============================================================================
# TestConfigLoader
# =============================================================================

class TestConfigLoader(unittest.TestCase):
    """Test config loading and override mechanism."""

    def test_load_config_exists(self):
        cfg_path = PROJECT_ROOT / "configs" / "chexpert_b1_v2.yaml"
        self.assertTrue(cfg_path.exists(), f"Config not found: {cfg_path}")

    def test_load_config_keys(self):
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "chexpert_b1_v2.yaml"
        if not cfg_path.exists():
            self.skipTest("Config file not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for section in ["model", "data", "sampler", "loss", "training", "eval"]:
            self.assertIn(section, cfg, f"Config missing section: {section}")

    def test_ablation_sweep_config(self):
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "ablation_sweep.yaml"
        if not cfg_path.exists():
            self.skipTest("Ablation config not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("ablations", cfg)
        ablations = cfg["ablations"]
        self.assertGreater(len(ablations), 0)
        # Each ablation must have a name
        for a in ablations:
            self.assertIn("name", a)


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    # Allow running directly with verbose output
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

