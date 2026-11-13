"""
tests/test_v4.py
================
Unit tests for the CheXpert V4 ("v4_gap") recipe per Section 9 specification:
1. convert_labels: blank->0 with weight 1, uncertain->soft target with s=0.5,
   ignore policy masks, shared by train and eval, eval excludes -1.
2. uslt_loss reduces to plain BCE when alpha=1, s=1, mask=1; no pos_weight used;
   alpha_c uses certain positives from the train split only.
3. DataLoader: random sampler, batch size 32, drop_last, no weighted sampler.
4. Stage A: backbone params have requires_grad False and BN modules are in eval();
   Stage B: only features[5:] unfrozen.
5. Model shapes at 240 and 320 (grid derived, not hardcoded); LabelAdaptiveHead
   identity-at-init, gradients reach label_emb; total params <= 12M.
6. No patient leakage across splits; metrics computed with prevalence/all-positive
   baselines are correct on a toy example.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import CHEXPERT_14_LABELS, convert_labels
from src.data.splits import make_patient_splits
from src.data.dataset import build_v4_dataloaders, CheXpertV4Dataset
from src.models.chexpert_v4 import CheXpertV4Model, LabelAdaptiveHead
from src.models.loss_v4 import uslt_loss, compute_alpha_c, USLTLossV4
from src.eval.protocol import evaluate_predictions, compute_ece


class TestConvertLabels:
    def test_blank_mapped_to_zero_with_weight_one(self):
        raw = torch.tensor([[float("nan")] * 14])
        y, u, blank, certain_pos, mask = convert_labels(raw, mode="train")
        assert (y == 0.0).all()
        assert (blank == 1.0).all()
        assert (mask == 1.0).all()
        assert (u == 0.0).all()
        assert (certain_pos == 0.0).all()

    def test_uncertain_mapped_to_soft_target(self):
        raw = torch.tensor([[-1.0] * 14])
        y, u, blank, certain_pos, mask = convert_labels(raw, mode="train")
        assert (y == 0.5).all()
        assert (u == 1.0).all()
        assert (mask == 1.0).all()
        assert (certain_pos == 0.0).all()

    def test_ignore_policy_masks_uncertain(self):
        cfg = {"labels": {"uncertain_policy": "ignore"}}
        raw = torch.tensor([[-1.0] * 14])
        y, u, blank, certain_pos, mask = convert_labels(raw, cfg=cfg, mode="train")
        assert (mask == 0.0).all()
        assert (y == 0.0).all()
        assert (u == 1.0).all()

    def test_eval_mode_excludes_uncertain(self):
        raw = torch.tensor([[1.0, 0.0, -1.0, float("nan")] + [0.0] * 10])
        y, u, blank, certain_pos, mask = convert_labels(raw, mode="eval")
        assert y[0, 0] == 1.0 and mask[0, 0] == 1.0   # pos
        assert y[0, 1] == 0.0 and mask[0, 1] == 1.0   # neg
        assert mask[0, 2] == 0.0 and y[0, 2] == 0.0   # unc EXCLUDED in eval!
        assert y[0, 3] == 0.0 and mask[0, 3] == 1.0   # blank -> true neg in eval!


class TestUSLTLoss:
    def test_reduces_to_plain_bce(self):
        B, C = 4, 14
        logits = torch.randn(B, C)
        y = torch.randint(0, 2, (B, C)).float()
        u = torch.zeros(B, C)
        blank = torch.zeros(B, C)
        mask = torch.ones(B, C)
        alpha = torch.ones(C)

        loss_uslt = uslt_loss(logits, y, u, blank, mask, alpha, lam_u=0.5, eps=0.1, blank_weight=1.0)
        expected_bce = F.binary_cross_entropy_with_logits(logits, y)
        assert torch.allclose(loss_uslt, expected_bce, atol=1e-5)

    def test_no_pos_weight_used(self):
        loss_mod = USLTLossV4(alpha=torch.ones(14))
        # Ensure pos_weight is not an attribute or buffer
        assert not hasattr(loss_mod, "pos_weight")
        assert "pos_weight" not in loss_mod.state_dict()

    def test_alpha_c_computed_from_certain_positives_only(self):
        # Create toy dataframe with certain pos (1), neg (0), unc (-1), and blank (NaN)
        data = {
            "No Finding": [1.0, 0.0, -1.0, float("nan")],
            "Cardiomegaly": [1.0, 1.0, -1.0, 0.0],
        }
        for lbl in CHEXPERT_14_LABELS:
            if lbl not in data:
                data[lbl] = [0.0, 0.0, 0.0, 0.0]
        df = pd.DataFrame(data)
        alpha, alpha_df = compute_alpha_c(df, labels=CHEXPERT_14_LABELS, beta=0.999)
        # No Finding has exactly 1 certain positive
        nf_row = alpha_df[alpha_df["class_name"] == "No Finding"].iloc[0]
        assert nf_row["n_certain_pos"] == 1
        # Cardiomegaly has exactly 2 certain positives
        card_row = alpha_df[alpha_df["class_name"] == "Cardiomegaly"].iloc[0]
        assert card_row["n_certain_pos"] == 2
        # Sum of alpha_c equals C (14)
        assert torch.allclose(alpha.sum(), torch.tensor(14.0), atol=1e-4)


class TestDataLoaderConstruction:
    def test_random_sampler_batch_size_drop_last(self):
        # Create toy dataframe
        rows = []
        for i in range(100):
            r = {"path_to_image": f"train/patient{i:05d}/study1/view1_frontal.jpg", "file_id": f"fid_{i}"}
            for lbl in CHEXPERT_14_LABELS:
                r[lbl] = 0.0
            rows.append(r)
        toy_df = pd.DataFrame(rows)
        cfg = {"batch_size": 32, "eval_batch_size": 64, "num_workers": 0, "sampler": "random"}
        train_loader, val_loader, test_loader = build_v4_dataloaders(toy_df, toy_df, toy_df, cfg=cfg)

        assert train_loader.batch_size == 32
        assert train_loader.drop_last is True
        # Random shuffle sampler is used
        assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)


class TestTrainingStages:
    def test_stage_a_freeze_and_eval_mode(self):
        cfg = {"input_size": 240, "max_params": 12_000_000}
        model = CheXpertV4Model(cfg)
        model.freeze_stage_a()

        # Backbone params must have requires_grad = False
        for p in model.backbone.parameters():
            assert not p.requires_grad

        # Backbone must be in eval mode
        assert not model.backbone.training
        # BN layers inside backbone must be in eval mode
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert not m.training

        # Heads must have requires_grad = True and be in training mode
        for m in [model.msfb, model.se, model.cbam, model.head]:
            assert m.training
            for p in m.parameters():
                assert p.requires_grad

    def test_stage_b_unfreeze_features_5_onwards(self):
        cfg = {"input_size": 240, "max_params": 12_000_000}
        model = CheXpertV4Model(cfg)
        model.unfreeze_stage_b(unfreeze_from=5)

        features = model.backbone.features
        # Blocks 0..4 must be frozen and in eval mode
        for i in range(5):
            assert not features[i].training
            for p in features[i].parameters():
                assert not p.requires_grad

        # Blocks 5..8 must be unfrozen and in train mode
        for i in range(5, len(features)):
            assert features[i].training
            for p in features[i].parameters():
                assert p.requires_grad


class TestModelArchitecture:
    def test_forward_shape_240(self):
        cfg = {"input_size": 240, "max_params": 12_000_000}
        model = CheXpertV4Model(cfg)
        x = torch.randn(2, 3, 240, 240)
        out = model(x)
        assert out.shape == (2, 14)

    def test_forward_shape_320(self):
        cfg = {"input_size": 320, "max_params": 12_000_000}
        model = CheXpertV4Model(cfg)
        x = torch.randn(2, 3, 320, 320)
        out = model(x)
        assert out.shape == (2, 14)

    def test_label_adaptive_head_identity_at_init(self):
        head = LabelAdaptiveHead(d=256, hidden=256, C=14, emb_dim=64, use_adapter=True, use_gate=False)
        f = torch.randn(3, 256)
        # Because W2 is initialized to zero, W2(act(W1(f))) == 0, so f stays f!
        adapter_out = f + head.W2(head.act(head.W1(f)))
        assert torch.allclose(adapter_out, f, atol=1e-6)

    def test_gradients_reach_label_emb(self):
        head = LabelAdaptiveHead(d=256, hidden=256, C=14, emb_dim=64, use_adapter=True, use_gate=True)
        f = torch.randn(2, 256, requires_grad=True)
        logits = head(f)
        loss = logits.sum()
        loss.backward()
        assert head.label_emb.weight.grad is not None
        assert not torch.allclose(head.label_emb.weight.grad, torch.zeros_like(head.label_emb.weight.grad))

    def test_total_params_under_12m(self):
        cfg = {"input_size": 240, "max_params": 12_000_000}
        model = CheXpertV4Model(cfg)
        total_p = sum(p.numel() for p in model.parameters())
        assert total_p <= 12_000_000


class TestSplitsAndEvaluation:
    def test_no_patient_leakage_across_splits(self):
        rows = []
        for i in range(100):
            # 2 images per patient
            for img in range(2):
                rows.append({
                    "path_to_image": f"train/patient{i:04d}/study1/view{img+1}_frontal.jpg",
                    "file_id": f"fid_{i}_{img}",
                    "local_filename": f"patient{i:04d}/study1/view{img+1}_frontal.png",
                    "No Finding": 0.0,
                })
        toy_df = pd.DataFrame(rows)
        tr, va, te, _ = make_patient_splits(toy_df, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=42)

        tr_pids = set(tr["patient_id"])
        va_pids = set(va["patient_id"])
        te_pids = set(te["patient_id"])

        assert len(tr_pids.intersection(va_pids)) == 0
        assert len(tr_pids.intersection(te_pids)) == 0
        assert len(va_pids.intersection(te_pids)) == 0

    def test_metrics_and_all_pos_baseline_on_toy_example(self):
        # 10 samples: 2 positive, 8 negative -> prevalence p = 0.2
        # Baseline all-positive F1: 2p / (1 + p) = 2*0.2 / 1.2 = 0.4 / 1.2 = 0.3333
        targets = np.zeros((10, 14))
        targets[:2, 0] = 1.0  # class 0 has 2 positives
        logits = np.zeros((10, 14))
        logits[:2, 0] = 2.0   # high logits for positives
        logits[2:, 0] = -2.0  # low logits for negatives
        masks = np.ones((10, 14))
        pids = np.array([f"p{i}" for i in range(10)])

        metrics_df, summary, _ = evaluate_predictions(
            logits=logits, targets=targets, masks=masks, patient_ids=pids, n_bootstraps=0
        )
        cls0 = metrics_df[metrics_df["class_name"] == "No Finding"].iloc[0]
        assert cls0["n_pos"] == 2
        assert cls0["n_neg"] == 8
        assert abs(cls0["prevalence"] - 0.2) < 1e-4
        assert abs(cls0["all_pos_f1"] - (2 * 0.2 / 1.2)) < 1e-3
        assert cls0["auroc"] == 1.0

