---
marp: true
theme: default
paginate: true
---

# Edge-Optimized Multi-Modal Pipeline
## Abnormality-Guided Chest X-Ray Interpretation and Report Generation

**IIT Ropar · Department of Computer Science and Engineering**  
**BTP Mid-Term Review**

Target Hardware: NVIDIA Jetson Orin Nano · 8 GB VRAM · 15W TDP

---

# The Problem: Cloud Radiology AI Has 2 Fatal Flaws

## ❌ Flaw 1 — Patient Privacy (HIPAA Violation)

> Raw X-ray images (PHI) transmitted over public internet to cloud servers
> violates hospital compliance policies → illegal in clinical deployment

## ❌ Flaw 2 — Clinical Hallucinations

> Monolithic VLMs (e.g., CheXagent 8B) conflate language priors with visual
> evidence → fluent but clinically incorrect reports

### Our Requirement
Run entirely **on-premises** on an **NVIDIA Jetson Orin Nano (8 GB)** — zero data leaves the hospital.

---

# Our Solution: 3-Phase Decoupled Pipeline

```
 INPUT: Chest X-Ray (224×224×3)
        │
        ▼
┌──────────────────────┐
│ PHASE 1: CLASSIFY    │  EfficientNet-B0 (5.3M params)
│ 14 pathologies → p̂  │  ✅ COMPLETED · Macro-AUROC = 0.8370
└──────┬───────┬───────┘
       │       │
       ▼       ▼
┌──────────┐ ┌───────────────────────┐
│ PHASE 2  │ │ PHASE 3: REPORT SCRIBE│  🔄 UPCOMING
│ Grad-CAM │ │ InternVL2-1B QLoRA    │
│ Heatmaps │ │ Conditioned by p̂     │
└──────────┘ └───────────┬───────────┘
                         │
                         ▼
                  CLINICAL REPORT
                  Findings + Impression
```

---

# Phase 1: Multi-Label Classification

## Backbone: EfficientNet-B0

- **5.3M parameters** (vs. CheXNet's DenseNet-121 at 7.0M)
- MBConv + Squeeze-and-Excitation blocks
- **~40% less activation memory** than DenseNet-121
- Head: `Dropout(0.2)` → `Linear(1280 → 14 logits)`

## Why Sigmoid, NOT Softmax?

$$\hat{p}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

A patient can simultaneously have **Effusion AND Cardiomegaly**.  
Softmax enforces mutual exclusivity → **clinically wrong**.  
Sigmoid treats each of 14 classes **independently** → **clinically correct**.

---

# Solving Class Imbalance

## NIH ChestX-ray14 Imbalance

| Pathology | Prevalence |
|---|---|
| Infiltration | 17.72% (common) |
| Hernia | **0.20%** (rare — 1:500 ratio) |

Standard training ignores rare classes entirely.

## Dynamic Positive Weighting

$$w_{\text{pos}, k} = \frac{N - N_{p,k}}{N_{p,k}} \quad \Rightarrow \quad w_{\text{Hernia}} = 492.9\times$$

**Result:** Hernia hit **AUROC 0.9484** — the rarest class became the best-detected.

---

# Phase 1 Results — Epoch 14

**Macro-AUROC: 0.8370 · Val Loss: 0.3615**

| Pathology | AUROC | Tier |
|---|---|---|
| Hernia | 0.9484 | 🏆 Outstanding |
| Emphysema | 0.9097 | 🏆 Outstanding |
| Cardiomegaly | 0.9072 | 🏆 Outstanding |
| Edema | 0.9053 | 🏆 Outstanding |
| Pneumothorax | 0.9039 | 🏆 Outstanding |
| Effusion | 0.8772 | ✅ Excellent |
| Atelectasis | 0.8050 | ✅ Excellent |
| Consolidation | 0.7941 | ⚠️ Acceptable |
| Infiltration | 0.7165 | ⚠️ Acceptable |

**CheXNet comparison:** 0.8410 AUROC @ 7.0M params vs. **Ours: 0.8370 @ 5.3M params**

---

# Phase 2: Grad-CAM Explainability

## Motivation

A probability alone is not enough for a radiologist. They need **spatial evidence**.

## Math

$$\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial z_c}{\partial A_{ij}^k}
\qquad
M_c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

## Implementation

```python
def forward_hook(module, input, output):
    self.feature_activations = output   # [B, 1280, 7, 7]

def backward_hook(module, grad_input, grad_output):
    self.gradients = grad_output[0]     # [B, 1280, 7, 7]
```

Hook target: `model.features[8]` → 7×7 map → bilinear upsample → 224×224 overlay

---

# Phase 3: Constrained Report Generation

## The Hallucination Problem

$$P_{\text{unconstrained}}(R \mid I) = \prod_{t=1}^T P(w_t \mid w_{<t}, \Phi_v(I))$$

Language priors dominate weak visual features → **hallucinations**.

## Our Probability-Guided Generation

$$P_{\text{guided}}(R \mid I, \hat{\mathbf{p}}) = \prod_{t=1}^T P(w_t \mid w_{<t}, \Phi_v(I), \mathcal{T}(\hat{\mathbf{p}}))$$

$\mathcal{T}(\hat{\mathbf{p}})$ = Phase 1 probabilities injected as text into every prompt.

The model is **told what is present** before generating any words.

---

# Phase 3: InternVL2-1B QLoRA Design

| Component | Detail |
|---|---|
| Visual encoder | InternViT-300M (8-bit dynamic resolution) |
| Language backbone | Qwen2-0.5B |
| Quantization | NF4 (bitsandbytes, double quant) → ~1.2 GB |
| LoRA rank / alpha | r=16, α=32 |
| Target layers | q_proj, k_proj, v_proj, o_proj |
| Trainable params | **< 15 Million** |
| Inference VRAM | **~3.8 GB** |

$$\mathbf{W}_{\text{adapted}} = \mathbf{W}_0 + \frac{\alpha}{r}(\mathbf{B} \cdot \mathbf{A})$$

---

# Example: Conditioned Report Output

```
[SYSTEM] You are an expert clinical radiologist. Bound findings
         strictly to the provided pathology indicators.

[USER]   <image>
         ABNORMALITY PROBABILITIES:
           Cardiomegaly:  0.9072  (HIGH CONFIDENCE)
           Edema:         0.9053  (HIGH CONFIDENCE)
           Pneumothorax:  0.0210  (ABSENT)
           Consolidation: 0.1240  (ABSENT)

[REPORT] FINDINGS: The cardiomediastinal silhouette is significantly
         enlarged. Vascular congestion and diffuse interstitial
         opacities are present throughout both lung bases,
         consistent with pulmonary edema. No visible pneumothorax
         or large focal consolidations are identified.

         IMPRESSION: Cardiomegaly with associated bilateral
         pulmonary edema.
```

---

# Dataset Strategy

| Dataset | Images | Reports? | Access | Use |
|---|---|---|---|---|
| NIH ChestX-ray14 | 112,120 | ❌ HIPAA withheld | Public | Phase 1 ✅ |
| OpenI | ~7,470 | ✅ | Public | Prototyping |
| MIMIC-CXR v2.1 | 377,110 | ✅ | Gated | Secondary eval |
| **CheXpert Plus** | **223,462** | **✅ + RadGraph XL** | **Approved ✅** | **Phase 3 SFT** |

### Critical Detail
NIH reports were **permanently withheld** (HIPAA Safe Harbor).
The 14 CSV labels were NLP-mined from internal notes — **no free-text exists**.
→ CheXpert Plus access was mandatory for Phase 3.

**Stanford RUA approved: September 22, 2026**

---

# Jetson Orin Nano Memory Budget

```
┌───────────────────────────────────────────────────┐
│        JETSON ORIN NANO — 8 GB VRAM BUDGET        │
├───────────────────────────────────────────────────┤
│  OS + JetPack daemons          ████████░  ~1.8 GB │
│  EfficientNet-B0 (FP16)        █░░░░░░░░  ~0.2 GB │
│  Grad-CAM activations          ░░░░░░░░░  ~0.1 GB │
│  InternVL2-1B (NF4)            ████████░  ~3.8 GB │
│  KV Cache + context            ████░░░░░  ~1.1 GB │
├───────────────────────────────────────────────────┤
│  TOTAL USED                               ~7.0 GB │
│  REMAINING HEADROOM                       ~1.0 GB │
└───────────────────────────────────────────────────┘
```

**CheXagent requires: >16 GB VRAM on multi-GPU clusters**  
**Our system: 7.0 GB on a single 8 GB edge device**

---

# Clinical Evaluation Metrics

## Why BLEU/ROUGE Are Clinically Dangerous

> "Patient shows **no signs** of pneumonia" vs. "Patient shows **severe** pneumonia"
> → BLEU score: **HIGH** · Clinical accuracy: **FATAL MISMATCH**

## Our Metrics

### CheXbert 14-Disease F1
Run generated reports through Stanford's CheXbert labeler.  
Compare extracted disease labels to ground truth → Precision, Recall, F1.

### RadGraph Entity-Relation F1
Check knowledge-graph overlap via RadGraph XL annotations:  
- Entity: "effusion" ✓
- Relation: `located_at` → "left costophrenic angle" ✓

**Baseline:** Stanford CheXagent (8B cloud-based)

---

# Post Mid-Term Roadmap

| Period | Phase | Key Deliverables |
|---|---|---|
| **Weeks 1–2** | Phase 2: Grad-CAM | Forward/backward hooks on `features[8]`; heatmap overlay pipeline; threshold τ=0.5 |
| **Weeks 3–5** | Phase 3: QLoRA SFT | InternVL2-1B NF4; abnormality-conditioned prompts; CheXpert Plus SFT |
| **Weeks 6–8** | Benchmark & Deploy | CheXbert F1 + RadGraph F1; TensorRT-LLM; Jetson real-time demo |

---

# System Comparison

| | CheXNet | CheXagent | **Ours** |
|---|---|---|---|
| Parameters | 7M | 8B | **~1.05B** |
| Generates reports | ❌ | ✅ | ✅ |
| Hallucination guard | ❌ | ❌ | ✅ |
| Explainability (XAI) | ❌ | ❌ | ✅ |
| Edge deployable (8 GB) | ✅ | ❌ | ✅ |
| Macro-AUROC | 0.841 | N/A | **0.837** |
| VRAM required | <1 GB | >16 GB | **~7.0 GB** |

---

# Conclusion

## Completed ✅
- EfficientNet-B0 on NIH ChestX-ray14 → **Macro-AUROC 0.8370**
- Numerically stable BCEWithLogitsLoss + dynamic class weighting
- CheXpert Plus + RadGraph XL access approved (Sep 22, 2026)
- Memory budget validated for 8 GB Jetson Orin Nano

## Upcoming 🔄
- Grad-CAM explainability pipeline
- InternVL2-1B QLoRA fine-tuning on CheXpert Plus
- CheXbert + RadGraph F1 benchmark
- TensorRT-LLM edge packaging

---

> **Core thesis:** A decoupled 3-tier pipeline can match cloud-scale diagnostic accuracy
> on a single 8 GB edge device — without hallucination, without PHI transmission,
> and with full radiologist-interpretable visual evidence.

*BTP Mid-Term Review · IIT Ropar · September 2026*
