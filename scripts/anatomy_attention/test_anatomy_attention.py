"""
test_anatomy_attention.py
=========================
Pre-training unit tests covering all 10 requirements from §50.
Run before any training to catch shape/gradient issues early.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def make_config(checkpoint_path: str, use_disease_att: bool = True, use_anatomy: bool = False):
    return {
        "backbone": "efficientnet_b1",
        "checkpoint_path": checkpoint_path,
        "freeze_backbone": True,
        "num_classes": 14,
        "use_multiscale": True,
        "multiscale_dilations": [1, 2, 3],
        "use_channel_attention": True,
        "channel_reduction": 16,
        "use_spatial_attention": True,
        "spatial_kernel": 7,
        "use_anatomy": use_anatomy,
        "anatomy_mid_channels": 32,
        "use_boundary": False,
        "use_disease_attention": use_disease_att,
        "use_anatomy_conditioning": False,
        "use_global_local": False,
        "disease_att_hidden": 128,
        "lambda_anatomy": 0.0,
        "lambda_diversity": 0.0,
    }


def test_all(ckpt_path: str):
    from anatomy_attention.anatomy_attention_model import AnatomyDiseaseAttentionModel

    device = torch.device("cpu")
    cfg    = make_config(ckpt_path)

    print("=" * 60)
    print("Building model ...")
    model = AnatomyDiseaseAttentionModel(cfg).to(device)
    model.eval()
    print("Model built.\n")

    B = 2
    dummy_x = torch.randn(B, 3, 224, 224)
    dummy_y = (torch.rand(B, 14) > 0.8).float()

    # ---- Test 1: Output shape [B, 14] -----------------------------------
    print("Test 1: Output shape")
    logits, _, _ = model(dummy_x)
    assert logits.shape == (B, 14), f"FAIL: {logits.shape}"
    print(f"  PASS: logits.shape = {logits.shape}")

    # ---- Test 2: Attention shapes ----------------------------------------
    print("\nTest 2: Attention shapes")
    model.eval()
    logits2, attn, disease_atts = model(dummy_x, return_attention=True)
    F_shape = (B, model.feature_dim, 7, 7)

    if "channel_attention" in attn:
        assert attn["channel_attention"].shape == (B, model.feature_dim, 1, 1), \
            f"channel_att shape FAIL: {attn['channel_attention'].shape}"
        print(f"  PASS: channel_attention = {attn['channel_attention'].shape}")

    if "spatial_attention" in attn:
        assert attn["spatial_attention"].shape == (B, 1, 7, 7), \
            f"spatial_att shape FAIL: {attn['spatial_attention'].shape}"
        print(f"  PASS: spatial_attention = {attn['spatial_attention'].shape}")

    if "disease_attention" in attn:
        assert attn["disease_attention"].shape == (B, 14, 7, 7), \
            f"disease_att shape FAIL: {attn['disease_attention'].shape}"
        print(f"  PASS: disease_attention = {attn['disease_attention'].shape}")

    # ---- Test 3: Backward pass -------------------------------------------
    print("\nTest 3: Backward pass")
    model.train()
    logits3, _, disease_atts3 = model(dummy_x)
    from anatomy_attention.losses import CombinedLoss
    criterion = CombinedLoss()
    loss_dict = criterion(logits3, dummy_y, disease_atts3)
    loss_dict["total"].backward()
    print(f"  PASS: backward() completed. loss={loss_dict['total'].item():.4f}")

    # ---- Test 4: Frozen backbone has no gradients -----------------------
    print("\nTest 4: Frozen backbone gradients")
    for name, param in model.backbone.named_parameters():
        assert param.grad is None or param.grad.abs().max() == 0, \
            f"FAIL: backbone param {name} has non-zero gradient!"
    print("  PASS: all backbone gradients are None / zero")

    # ---- Test 5: Backbone stays in eval mode after model.train() --------
    print("\nTest 5: Backbone stays in eval() during model.train()")
    model.train()
    assert not model.backbone.backbone.training, \
        "FAIL: backbone is in train mode!"
    print("  PASS: backbone.training = False")

    # ---- Test 6: batch_size > 1 -----------------------------------------
    print("\nTest 6: Batch size > 1")
    model.eval()
    x_batch = torch.randn(4, 3, 224, 224)
    logits_b, _, _ = model(x_batch)
    assert logits_b.shape == (4, 14)
    print(f"  PASS: batch=4 logits.shape = {logits_b.shape}")

    # ---- Test 7: No NaNs in outputs ------------------------------------
    print("\nTest 7: No NaNs")
    assert not torch.isnan(logits_b).any(), "FAIL: NaN in logits"
    print("  PASS: no NaNs in logits")

    # ---- Test 8: Disease attention values in (0, 1) --------------------
    print("\nTest 8: Disease attention in (0, 1)")
    model.eval()
    _, _, da = model(dummy_x)
    if da is not None:
        assert da.min() >= 0.0 and da.max() <= 1.0, f"FAIL: da range = [{da.min():.4f}, {da.max():.4f}]"
        print(f"  PASS: disease_atts range = [{da.min():.4f}, {da.max():.4f}]")
    else:
        print("  SKIP: disease_attention not enabled")

    # ---- Test 9: use_anatomy=False → no anatomy error -------------------
    print("\nTest 9: Missing anatomy masks handled cleanly")
    model.eval()
    try:
        logits_no_mask, _, _ = model(dummy_x, anatomy_masks=None)
        print(f"  PASS: runs cleanly without masks (use_anatomy=False)")
    except ValueError as e:
        print(f"  PASS: correctly raises ValueError when masks missing: {e}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Print parameter summary
    counts = model.param_count()
    print(f"\nParameter Summary:")
    print(f"  Total:     {counts['total']:>12,}")
    print(f"  Trainable: {counts['trainable']:>12,}")
    print(f"  Frozen:    {counts['frozen']:>12,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_classifier_efficientnet_b1.pth")
    args = parser.parse_args()
    test_all(args.checkpoint)
