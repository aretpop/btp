"""
scripts/test_batch_2.py
"""
import sys, torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))

from train_vlm_internvl2 import load_classifier, load_vlm, preprocess_internvl2, build_conversation, SYSTEM_PROMPT, REPORT_INDEX, CACHE_DIR
from chexpert_report_dataset import CheXpertReportDataset
import redivis

device = torch.device('cuda')
classifier = load_classifier(device)
table = redivis.user('aimi').dataset('chexpert_plus:5yyj:v1_0').table('png_train:s6cj')

train_ds = CheXpertReportDataset(
    report_index_path=str(REPORT_INDEX),
    classifier_model=classifier,
    cache_dir=str(CACHE_DIR),
    redivis_table=table,
    epoch_size=10
)

vlm, tok = load_vlm(device)
vlm.eval()
vlm.language_model.train()

samples = [train_ds[0], train_ds[1]]

dtype = torch.bfloat16
pixel_values_list = []
input_ids_list = []
labels_list = []

IMG_START = "<img>"
IMG_END   = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"
num_image_tokens = getattr(vlm, "num_image_token", 256)
image_tokens = IMG_START + IMG_CONTEXT * num_image_tokens + IMG_END
assistant_token = tok.encode("<|im_start|>assistant", add_special_tokens=False)

for s in samples:
    pv_xray = preprocess_internvl2(s["image_pil"], device, dtype)
    pv_attn = preprocess_internvl2(s["attn_grid_pil"], device, dtype)
    pixel_values_list.append(torch.cat([pv_xray, pv_attn], dim=0))

    question, _ = build_conversation(s["prob_text"], tok)
    full_question = question.replace("<image>", image_tokens, 2)
    full_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{full_question}<|im_end|>\n"
        f"<|im_start|>assistant\n{s['report_text']}<|im_end|>"
    )
    enc = tok(full_text, return_tensors="pt", max_length=768, truncation=True, padding=False)
    inp_ids = enc["input_ids"][0]
    lbls = inp_ids.clone()
    s_len = len(inp_ids)
    mask_until = s_len
    for i in range(s_len - len(assistant_token)):
        if inp_ids[i:i+len(assistant_token)].tolist() == assistant_token:
            mask_until = i + len(assistant_token) + 1
            break
    lbls[:mask_until] = -100
    input_ids_list.append(inp_ids)
    labels_list.append(lbls)

batch_pv = torch.cat(pixel_values_list, dim=0)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
max_len = max(len(ids) for ids in input_ids_list)

batch_input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long, device=device)
batch_attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long, device=device)
batch_labels = torch.full((len(samples), max_len), -100, dtype=torch.long, device=device)

for b_idx in range(len(samples)):
    cur_len = len(input_ids_list[b_idx])
    batch_input_ids[b_idx, :cur_len] = input_ids_list[b_idx].to(device)
    batch_attention_mask[b_idx, :cur_len] = 1
    batch_labels[b_idx, :cur_len] = labels_list[b_idx].to(device)

image_flags = torch.ones(batch_pv.shape[0], dtype=torch.long, device=device)

inputs = {
    "pixel_values": batch_pv,
    "input_ids": batch_input_ids,
    "attention_mask": batch_attention_mask,
    "image_flags": image_flags,
    "labels": batch_labels
}

print("Inputs built! batch_input_ids shape:", batch_input_ids.shape)
print("Sample 0 valid labels:", (batch_labels[0] != -100).sum().item())
print("Sample 1 valid labels:", (batch_labels[1] != -100).sum().item())
out = vlm(**inputs)
print("Loss:", out.loss)
if out.loss is not None and not torch.isnan(out.loss):
    print("Loss value:", out.loss.item())
else:
    print("Checking logits for nan/inf...")
    print("logits has nan:", torch.isnan(out.logits).any().item())
    print("logits has inf:", torch.isinf(out.logits).any().item())
out.loss.backward()
print("Backward completed successfully! Peak VRAM:", torch.cuda.max_memory_allocated()/(1024**3), "GB")
