"""
scripts/train_vlm_internvl2.py
================================
QLoRA fine-tuning of InternVL2-1B for chest X-ray report generation.

Design:
  - Base model   : OpenGVLab/InternVL2-1B (4-bit NF4 quantization via bitsandbytes)
  - Fine-tuning  : LoRA (r=8, alpha=16) on LLM q/v projections only
  - Frozen       : InternVL2 vision encoder (intern_vit) + our classifier backbone
  - Input        : [X-ray image] + [Attention-map grid image] + classifier probs text
  - Target       : section_findings + section_impression (report)
  - Batching     : batch_size=2, grad_accum=4 (effective batch = 8)
  - Sampling     : 8,000 uniform random samples/epoch (same as transfer training)
  - VRAM budget  : ~5.0–5.5 GB on RTX 4060 (leaves 2.5–3.0 GB headroom)
  - Memory guards: periodic empty_cache(), gc.collect(), zero_grad(set_to_none=True)

Usage:
  python scripts/train_vlm_internvl2.py                # Auto-resumes from latest epoch
  python scripts/train_vlm_internvl2.py --smoke-test   # 6 steps dummy test
  python scripts/train_vlm_internvl2.py --fresh        # Start from epoch 1
"""

import os, sys, gc, time, logging, argparse
os.environ.setdefault("REDIVIS_API_TOKEN", "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))  # avoid collision with HF 'datasets' pkg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_CLASSIFIER = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / \
                  "stageB_efficientnet_b1_attention.pth"
REPORT_INDEX    = PROJECT_ROOT / "data" / "chexpert_subset" / "train_report_index.csv"
CACHE_DIR       = PROJECT_ROOT / "data" / "chexpert_cache"
VAL_CSV         = PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv"
VAL_CACHE       = PROJECT_ROOT / "data" / "chexpert_val_cache"
CKPT_OUT_DIR    = PROJECT_ROOT / "checkpoints" / "vlm_report"
CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

VLM_MODEL_ID   = "OpenGVLab/InternVL2-1B"
EPOCH_SIZE     = 8_000
EPOCHS         = 10
BATCH_SIZE     = 2
GRAD_ACCUM     = 4        # effective batch = 2 × 4 = 8
LR_LORA        = 2e-4
LR_PROJ        = 1e-3     # higher LR for projector if unfreezing
MAX_NEW_TOKENS = 256
MAX_SEQ_LEN    = 1024     # 2 images × 256 tokens = 512 img tokens alone; prompt is ~774 tokens


# ── Helpers ────────────────────────────────────────────────────────────────────
def gb(b): return b / (1024 ** 3)

def mem():
    return gb(torch.cuda.memory_allocated()), gb(torch.cuda.max_memory_allocated())

def suppress_flash_attn():
    import sys, types, importlib.util
    if "flash_attn" not in sys.modules:
        m = types.ModuleType("flash_attn")
        m.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
        m.__version__ = "0.0.0"
        sys.modules["flash_attn"] = m
        sys.modules["flash_attn.flash_attn_interface"] = m


def find_latest_checkpoint(ckpt_dir: Path):
    """Find the latest completed epoch checkpoint in ckpt_dir."""
    candidates = [p for p in ckpt_dir.glob("lora_ep*") if p.is_dir()]
    if not candidates:
        return None, 1, 0, float("inf")

    def get_epoch(p):
        try:
            return int(p.name.replace("lora_ep", ""))
        except ValueError:
            return 0

    candidates = sorted(candidates, key=get_epoch, reverse=True)
    latest = candidates[0]
    meta_path = latest / "meta.pth"
    if meta_path.exists():
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            epoch = meta.get("epoch", get_epoch(latest))
            global_step = meta.get("global_step", 0)
            loss = meta.get("loss", float("inf"))
            return latest, epoch + 1, global_step, loss
        except Exception as e:
            log.warning(f"Failed to read meta.pth from {latest}: {e}")
            return latest, get_epoch(latest) + 1, 0, float("inf")
    return latest, get_epoch(latest) + 1, 0, float("inf")


# ── Step 1: Load our Stage B classifier (frozen, self-contained) ───────────────
def _build_transfer_model(backbone_name: str, num_classes: int):
    """
    Reconstructs CheXpertTransferModel without importing from train_phase1_classifier
    (which has a top-level mlflow import that fails in envs without mlflow).
    """
    import copy, torch.nn as nn, torch.nn.functional as F
    from torchvision import models
    from train_phase3_regional_v3 import (
        MultiScaleFeatureBlock, ChannelAttention, SpatialAttention,
        DiseaseAttentionGenerator, BLOCK5_CHANNELS, BLOCK8_CHANNELS,
    )

    class _ChestXRayModel(nn.Module):
        def __init__(self, backbone_name, num_classes):
            super().__init__()
            backbone = getattr(models, backbone_name)(weights=None)
            self.backbone = backbone
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
        def forward(self, x):
            return self.backbone(x)

    class _CheXpertTransferModel(nn.Module):
        def __init__(self, nih_backbone, backbone_name, use_attention, num_classes):
            super().__init__()
            self.backbone_name = backbone_name
            self.num_classes   = num_classes
            self.use_attention = use_attention
            self.backbone      = copy.deepcopy(nih_backbone)
            for p in self.backbone.parameters():
                p.requires_grad = False
            b8_ch   = BLOCK8_CHANNELS[backbone_name]
            b5_ch   = BLOCK5_CHANNELS[backbone_name]
            feat_dim = 256
            if use_attention:
                self.refine      = MultiScaleFeatureBlock(b8_ch, feat_dim)
                self.channel_att = ChannelAttention(feat_dim)
                self.spatial_att = SpatialAttention()
                self.disease_att = DiseaseAttentionGenerator(b5_ch, num_classes, att_norm="sigmoid")
                self.classifiers = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(num_classes)])

        def _get_features(self, x):
            blocks = list(self.backbone.backbone.features) \
                if hasattr(self.backbone, "backbone") else list(self.backbone.features)
            z, f5 = x, None
            for i, block in enumerate(blocks):
                z = block(z)
                if i == 5:
                    f5 = z
            return f5, z

        def forward(self, x, return_attention=False):
            f5, f8 = self._get_features(x)
            f_ref    = self.refine(f8)
            f_ref    = self.channel_att(f_ref)
            f_ref    = self.spatial_att(f_ref)
            f_shared = F.interpolate(f_ref, size=(14, 14), mode="bilinear", align_corners=False)
            disease_maps = self.disease_att(f5)
            H, W = f_shared.shape[2], f_shared.shape[3]
            f_flat = f_shared.flatten(2)
            d_flat = disease_maps.flatten(2)
            z = torch.bmm(d_flat, f_flat.transpose(1, 2)) / (H * W)
            logits = torch.cat([self.classifiers[d](z[:, d, :]) for d in range(self.num_classes)], dim=1)
            if return_attention:
                return logits, disease_maps
            return logits

    nih = _ChestXRayModel(backbone_name, 14)
    return _CheXpertTransferModel(nih, backbone_name, use_attention=True, num_classes=num_classes)


def load_classifier(device):
    NUM_CHEXPERT = 11
    model = _build_transfer_model("efficientnet_b1", NUM_CHEXPERT)
    ckpt  = torch.load(CKPT_CLASSIFIER, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    log.info(f"Classifier loaded: {sum(p.numel() for p in model.parameters()):,} params (frozen)")
    return model


# ── Step 2: Load InternVL2-1B with 4-bit QLoRA ─────────────────────────────
def load_vlm(device, resume_path=None):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel

    suppress_flash_attn()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading InternVL2-1B in 4-bit NF4...")

    # Monkey-patch to fix transformers 4.44 + accelerate 1.14 + bnb 0.50
    import transformers as _tf
    _orig_to = _tf.PreTrainedModel.to
    def _bnb_safe_to(self, *args, **kwargs):
        try:
            return _orig_to(self, *args, **kwargs)
        except ValueError as e:
            if "4-bit" in str(e) or "8-bit" in str(e):
                log.debug("Suppressed BNB .to() ValueError (expected during dispatch)")
                return self
            raise
    _tf.PreTrainedModel.to = _bnb_safe_to

    try:
        vlm = AutoModel.from_pretrained(
            VLM_MODEL_ID,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    finally:
        _tf.PreTrainedModel.to = _orig_to

    tok = AutoTokenizer.from_pretrained(
        VLM_MODEL_ID, trust_remote_code=True, use_fast=False
    )

    cur, peak = mem()
    log.info(f"4-bit model loaded | VRAM: {cur:.2f} GB allocated, {peak:.2f} GB peak")

    # Freeze vision encoder explicitly
    if hasattr(vlm, "vision_model"):
        for p in vlm.vision_model.parameters():
            p.requires_grad = False
        log.info("Vision encoder frozen")

    # Prepare for k-bit training (handles quantised layer hooks)
    vlm = prepare_model_for_kbit_training(vlm, use_gradient_checkpointing=True)

    # ── KEY: apply LoRA to the INNER language model only ──────────────────────
    if resume_path is not None and Path(resume_path).exists():
        log.info(f"Resuming LoRA adapter from {resume_path}...")
        vlm.language_model = PeftModel.from_pretrained(
            vlm.language_model, str(resume_path), is_trainable=True
        )
        trainable = sum(p.numel() for p in vlm.language_model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in vlm.parameters())
        log.info(f"Resumed trainable params: {trainable:,} || all params: {total:,} || "
                 f"trainable%: {100*trainable/total:.4f}")
    else:
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if hasattr(vlm, "language_model"):
            vlm.language_model = get_peft_model(vlm.language_model, lora_cfg)
            trainable = sum(p.numel() for p in vlm.language_model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in vlm.parameters())
            log.info(f"trainable params: {trainable:,} || all params: {total:,} || "
                     f"trainable%: {100*trainable/total:.4f}")
        else:
            vlm = get_peft_model(vlm, lora_cfg)
            vlm.print_trainable_parameters()

    # ── CRITICAL: set img_context_token_id ────────────────────────────────────
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    vlm.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    log.info(f"img_context_token_id set to: {vlm.img_context_token_id}")

    # ── Cast vision encoder + MLP1 + input embeddings to bfloat16 LAST ───────
    for submod_name in ["vision_model", "mlp1"]:
        submod = getattr(vlm, submod_name, None)
        if submod is not None:
            for p in submod.parameters():
                p.data = p.data.to(torch.bfloat16)
            for b in submod.buffers():
                b.data = b.data.to(torch.bfloat16)

    if hasattr(vlm, "language_model"):
        emb = vlm.language_model.get_input_embeddings()
        if emb is not None:
            emb.to(torch.bfloat16)
            log.info("language_model input embeddings cast to bfloat16")

    log.info("vision_model + mlp1 + embeddings dtype-fixed to bfloat16 (post-kbit-prep)")

    # ── Move all remaining CPU tensors to CUDA ────────────────────────────────
    log.info("Migrating remaining CPU tensors to CUDA...")
    n_moved_buf, n_moved_par = 0, 0

    for mod_name, module in vlm.named_modules():
        for buf_key in list(module._buffers.keys()):
            buf = module._buffers[buf_key]
            if buf is not None and buf.device.type == "cpu":
                module._buffers[buf_key] = buf.to("cuda")
                n_moved_buf += 1

        for par_key in list(module._parameters.keys()):
            par = module._parameters[par_key]
            if par is not None and par.device.type == "cpu":
                if not hasattr(par, "quant_state"):
                    module._parameters[par_key] = torch.nn.Parameter(
                        par.data.to("cuda"), requires_grad=par.requires_grad
                    )
                    n_moved_par += 1

    log.info(f"Migrated {n_moved_buf} buffers + {n_moved_par} params to CUDA")
    cur, peak = mem()
    log.info(f"VRAM after CUDA migration: {cur:.2f} GB allocated, {peak:.2f} GB peak")

    return vlm, tok


# ── Step 3: InternVL2 image preprocessing ─────────────────────────────────────
def preprocess_internvl2(pil_img: Image.Image, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Preprocess a PIL image to InternVL2 pixel_values format [1, 3, 448, 448]."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(pil_img).unsqueeze(0).to(device=device, dtype=dtype)


# ── Step 4: Build conversation prompt & training batches ──────────────────────
SYSTEM_PROMPT = (
    "You are an expert radiologist AI assistant. "
    "You are provided with a single frontal chest X-ray image, disease attention maps, and pre-computed classifier findings. "
    "This is a baseline examination with NO prior study or comparison available. "
    "Describe ONLY current findings visible on this radiograph. "
    "Do NOT mention prior examinations, interval changes, or previous studies. "
    "Generate a complete, clinically accurate radiology report with FINDINGS and IMPRESSION sections."
)

def build_conversation(prob_text: str, tok) -> tuple:
    """Build InternVL2 conversation format with two images."""
    question = (
        "<image>\n<image>\n"
        f"{prob_text}\n\n"
        "Based on the chest X-ray and the disease attention maps above, "
        "generate a complete radiology report with FINDINGS and IMPRESSION sections. "
        "This is a standalone baseline study with NO prior comparison. "
        "Describe ONLY current findings without referencing prior examinations."
    )
    return question, None


def build_training_batch(
    samples: list,
    vlm,
    tok,
    device,
    max_seq_len: int = MAX_SEQ_LEN
) -> dict:
    """
    Build padded model inputs for a batch of training samples (B >= 1).
    Each sample contains:
        image_pil, attn_grid_pil, prob_text, report_text
    """
    dtype = torch.bfloat16

    IMG_START = "<img>"
    IMG_END   = "</img>"
    IMG_CONTEXT = "<IMG_CONTEXT>"
    num_image_tokens = getattr(vlm, "num_image_token", 256)
    image_tokens = IMG_START + IMG_CONTEXT * num_image_tokens + IMG_END
    assistant_token = tok.encode("<|im_start|>assistant", add_special_tokens=False)

    pixel_values_list = []
    input_ids_list = []
    labels_list = []

    for s in samples:
        # Preprocess both images: [1, 3, 448, 448] each
        pv_xray = preprocess_internvl2(s["image_pil"], device, dtype)
        pv_attn = preprocess_internvl2(s["attn_grid_pil"], device, dtype)
        pixel_values_list.append(torch.cat([pv_xray, pv_attn], dim=0))     # [2, 3, 448, 448]

        question, _ = build_conversation(s["prob_text"], tok)
        full_question = question.replace("<image>", image_tokens, 2)
        full_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{full_question}<|im_end|>\n"
            f"<|im_start|>assistant\n{s['report_text']}<|im_end|>"
        )

        enc = tok(
            full_text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        inp_ids = enc["input_ids"][0]
        lbls = inp_ids.clone()
        seq_len = len(inp_ids)
        mask_until = seq_len
        for i in range(seq_len - len(assistant_token)):
            if inp_ids[i:i+len(assistant_token)].tolist() == assistant_token:
                mask_until = i + len(assistant_token) + 1   # +1 for newline
                break
        lbls[:mask_until] = -100

        input_ids_list.append(inp_ids)
        labels_list.append(lbls)

    batch_pv = torch.cat(pixel_values_list, dim=0)   # [2 * B, 3, 448, 448]
    b_size = len(samples)
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    batch_input_ids = torch.full((b_size, max_len), pad_id, dtype=torch.long, device=device)
    batch_attention_mask = torch.zeros((b_size, max_len), dtype=torch.long, device=device)
    batch_labels = torch.full((b_size, max_len), -100, dtype=torch.long, device=device)

    for b_idx in range(b_size):
        cur_len = len(input_ids_list[b_idx])
        batch_input_ids[b_idx, :cur_len] = input_ids_list[b_idx].to(device)
        batch_attention_mask[b_idx, :cur_len] = 1
        batch_labels[b_idx, :cur_len] = labels_list[b_idx].to(device)

    image_flags = torch.ones(batch_pv.shape[0], dtype=torch.long, device=device)

    return {
        "pixel_values":   batch_pv,
        "input_ids":      batch_input_ids,
        "attention_mask": batch_attention_mask,
        "image_flags":    image_flags,
        "labels":         batch_labels,
    }


def build_training_input(
    pil_img: Image.Image,
    attn_grid_pil: Image.Image,
    prob_text: str,
    report_text: str,
    vlm,
    tok,
    device,
) -> dict:
    """Backward compatibility wrapper for single sample evaluation."""
    sample = {
        "image_pil": pil_img,
        "attn_grid_pil": attn_grid_pil,
        "prob_text": prob_text,
        "report_text": report_text,
    }
    return build_training_batch([sample], vlm, tok, device, max_seq_len=MAX_SEQ_LEN)


# ── Step 5: Training loop ──────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda")

    batch_size = args.batch_size
    grad_accum = args.grad_accum
    epochs = args.epochs
    epoch_size = EPOCH_SIZE if not args.smoke_test else 6

    # 1. Load classifier (frozen)
    log.info("=== Loading Stage B Classifier ===")
    classifier = load_classifier(device)
    cur, _ = mem()
    log.info(f"Classifier VRAM: {cur:.2f} GB")

    # 2. Check for resume checkpoint
    latest_ckpt, start_epoch, global_step, best_loss = find_latest_checkpoint(CKPT_OUT_DIR)
    if args.fresh:
        latest_ckpt = None
        start_epoch = 1
        global_step = 0
        best_loss = float("inf")

    # 3. Load VLM with QLoRA
    log.info("\n=== Loading InternVL2-1B + QLoRA ===")
    if latest_ckpt is not None:
        log.info(f"Resume detected: Loading adapter from {latest_ckpt.name}")
        log.info(f"Resuming at Epoch {start_epoch} | Global Step {global_step} | Best Loss: {best_loss:.4f}")
        vlm, tok = load_vlm(device, resume_path=latest_ckpt)
    else:
        log.info("Starting fresh training (Epoch 1)...")
        vlm, tok = load_vlm(device, resume_path=None)

    cur, _ = mem()
    log.info(f"VLM + classifier total VRAM: {cur:.2f} GB")

    # 4. Set up dataset
    if not args.smoke_test:
        if not REPORT_INDEX.exists():
            log.error(f"Report index not found: {REPORT_INDEX}")
            log.error("Run: python scripts/build_report_index.py first")
            sys.exit(1)

        import redivis
        dataset = redivis.user("aimi").dataset("chexpert_plus:5yyj:v1_0")
        table   = dataset.table("png_train:s6cj")

        from chexpert_report_dataset import CheXpertReportDataset
        train_ds = CheXpertReportDataset(
            report_index_path=str(REPORT_INDEX),
            classifier_model=classifier,
            cache_dir=str(CACHE_DIR),
            redivis_table=table,
            epoch_size=epoch_size,
        )
        # Advance dataset random state for completed epochs
        for ep in range(1, start_epoch):
            log.info(f"Advancing dataset random state for completed Epoch {ep}...")
            train_ds.on_epoch_end()
    else:
        log.info("=== SMOKE TEST MODE (6 samples, batch_size=2) ===")

    # 5. Optimizer: LoRA params are in vlm.language_model (PeftModel)
    lm = vlm.language_model
    lora_params = [p for n, p in lm.named_parameters()
                   if "lora_" in n and p.requires_grad]
    log.info(f"Optimizer tracking {len(lora_params)} LoRA parameter tensors")
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": LR_LORA}],
        weight_decay=0.01,
    )

    # 6. Scheduler: linear warmup then cosine OneCycleLR
    from torch.optim.lr_scheduler import OneCycleLR
    steps_per_epoch = (epoch_size + (batch_size * grad_accum) - 1) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR_LORA,
        total_steps=max(total_steps, 1),
        pct_start=0.05
    )

    # Fast-forward scheduler if resuming
    if global_step > 0:
        steps_to_advance = min(global_step, max(total_steps - 1, 0))
        log.info(f"Fast-forwarding scheduler by {steps_to_advance} steps (out of {total_steps} total steps)...")
        for _ in range(steps_to_advance):
            scheduler.step()
        log.info(f"Resumed learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Outer vlm in eval (frozen vision encoder); inner lm adapter in train mode
    vlm.eval()
    lm.train()
    optimizer.zero_grad(set_to_none=True)

    log.info(f"\n=== Training Configuration ===")
    log.info(f"Epochs: {start_epoch} to {epochs} (Total target: {epochs})")
    log.info(f"Samples/epoch: {epoch_size} | Batch size: {batch_size} | Grad accum: {grad_accum}")
    log.info(f"Effective batch size: {batch_size * grad_accum} | Steps/epoch: {steps_per_epoch}")
    log.info(f"Peak VRAM ceiling: 8.00 GB (Target: ~5.0–5.5 GB with periodic empty_cache)")

    num_batches = (epoch_size + batch_size - 1) // batch_size

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        samples_processed = 0
        epoch_start = time.time()

        for batch_i in range(num_batches):
            b_start = batch_i * batch_size
            b_end   = min(b_start + batch_size, epoch_size)
            current_b_size = b_end - b_start

            # Get batch samples
            if args.smoke_test:
                from chexpert_report_dataset import format_classifier_text, CHEXPERT_LABELS
                batch_samples = []
                for _ in range(current_b_size):
                    batch_samples.append({
                        "image_pil": Image.new("RGB", (224, 224), 128),
                        "attn_grid_pil": Image.new("RGB", (224, 224), 64),
                        "prob_text": format_classifier_text(np.random.rand(11), CHEXPERT_LABELS),
                        "report_text": "FINDINGS:\nCardiac silhouette is normal.\n\nIMPRESSION:\nNo acute cardiopulmonary disease.",
                    })
            else:
                batch_samples = [train_ds[idx] for idx in range(b_start, b_end)]

            # Build batch inputs
            try:
                inputs = build_training_batch(batch_samples, vlm, tok, device, max_seq_len=MAX_SEQ_LEN)
            except Exception as e:
                log.warning(f"Batch {batch_i} input build failed: {e}")
                continue

            # Forward & Backward
            try:
                out  = vlm(**inputs)
                loss = out.loss / grad_accum
                loss.backward()
                batch_loss = out.loss.item()
                epoch_loss += batch_loss * current_b_size
                samples_processed += current_b_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.error(f"OOM at batch {batch_i}! Skipping and clearing cache.")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            finally:
                # Explicitly release references to prevent CUDA caching allocator retention
                del inputs
                if "out" in locals():
                    del out
                if "loss" in locals():
                    del loss

            # Gradient step
            if (batch_i + 1) % grad_accum == 0 or (batch_i + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 25 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    cur, peak = mem()
                    curr_lr = optimizer.param_groups[0]["lr"]
                    avg_running = epoch_loss / max(samples_processed, 1)
                    log.info(
                        f"Ep {epoch} | Step {global_step} | "
                        f"Loss {avg_running:.4f} | LR {curr_lr:.2e} | "
                        f"VRAM {cur:.2f}GB (peak {peak:.2f}GB)"
                    )

        avg_loss = epoch_loss / max(samples_processed, 1)
        elapsed  = time.time() - epoch_start
        log.info(f"\n── Epoch {epoch}/{epochs} complete | Loss: {avg_loss:.4f} | {elapsed/60:.1f} min")

        # Save LoRA adapter checkpoint
        save_dir = CKPT_OUT_DIR / f"lora_ep{epoch:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        lm.save_pretrained(str(save_dir))
        torch.save({
            "epoch": epoch,
            "loss": avg_loss,
            "global_step": global_step,
            "best_loss": best_loss,
        }, save_dir / "meta.pth")
        log.info(f"Saved checkpoint → {save_dir}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            lm.save_pretrained(str(CKPT_OUT_DIR / "lora_best"))
            torch.save({
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": global_step,
                "best_loss": best_loss,
            }, CKPT_OUT_DIR / "lora_best" / "meta.pth")
            log.info(f"  ★ New best loss: {best_loss:.4f} (saved to lora_best)")

        # Clear cache at epoch end
        torch.cuda.empty_cache()
        gc.collect()

        if not args.smoke_test:
            train_ds.on_epoch_end()

    log.info(f"\n=== Training Complete | Best Loss: {best_loss:.4f} ===")
    log.info(f"Best checkpoint: {CKPT_OUT_DIR / 'lora_best'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of InternVL2-1B for report generation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 6 samples dummy test to verify pipeline")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh training ignoring existing checkpoints")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device forward batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4, effective batch: 8)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs (default: 10)")
    args = parser.parse_args()
    train(args)
