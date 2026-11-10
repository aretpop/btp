# Baseline Clinical Efficacy Evaluation for Chest X-Ray Report Generation

This repository provides a reproducible evaluation pipeline to measure the baseline Clinical Efficacy (CE) of general-purpose Vision-Language Models (VLMs) on the IU X-Ray dataset.

## Dataset

The dataset is expected to be located in the `IU-Xray` directory at the project root.
- Images: `IU-Xray/image/...`
- Splits: `train.jsonl`, `val.jsonl`, `test.jsonl`

## Installation

1. Create a Python environment (Python 3.9 - 3.11 recommended).
2. Install PyTorch appropriate for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/).
3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow & Commands

The pipeline is structured into independent scripts that should be run sequentially.

### 1. Environment Diagnostics
Check hardware, GPU availability, and model feasibility.
```bash
python scripts/check_environment.py
```

### 2. Dataset Validation
Validate the integrity of the dataset splits.
```bash
python scripts/validate_dataset.py --split all
```

### 3. Sample Inspection
Verify images load correctly and reports format properly. Use `--show` to save a plot to `experiments/`.
```bash
python scripts/inspect_sample.py --index 0 --show
```

### 4. Run Smoke Test (Generation)
Test the generation pipeline on a small subset (default 8 samples).
```bash
python scripts/generate_reports.py --smoke
```
*Note: This creates a timestamped folder under `experiments/` which you'll need for subsequent steps.*

### 5. Label Extraction
Extract clinical labels (using CheXpert-style regex labeler by default) from both the generated and reference reports. Replace `<EXPERIMENT_DIR>` with the folder created in step 4.
```bash
python scripts/run_clinical_labeler.py --exp_dir experiments/<EXPERIMENT_DIR>
```

### 6. Calculate Metrics
Compute overall and per-pathology Precision, Recall, and F1.
```bash
python scripts/calculate_metrics.py --exp_dir experiments/<EXPERIMENT_DIR>
```

### 7. Error Analysis
Identify exact samples where models missed a finding or hallucinated a false positive.
```bash
python scripts/error_analysis.py --exp_dir experiments/<EXPERIMENT_DIR>
```

### 8. Generate Plots
Create publication-ready visualizations (saved in the `plots/` subdirectory).
```bash
python scripts/plot_results.py --exp_dir experiments/<EXPERIMENT_DIR>
```

## Running the Full Evaluation
After verifying the smoke test succeeds, run the full pipeline:
```bash
python scripts/generate_reports.py
python scripts/run_clinical_labeler.py --exp_dir experiments/<NEW_EXPERIMENT_DIR>
python scripts/calculate_metrics.py --exp_dir experiments/<NEW_EXPERIMENT_DIR>
python scripts/error_analysis.py --exp_dir experiments/<NEW_EXPERIMENT_DIR>
python scripts/plot_results.py --exp_dir experiments/<NEW_EXPERIMENT_DIR>
```

## Configuration
Modify `configs/baseline.yaml` to change models, prompt, generation settings, or labeler configuration.

## Known Limitations
* **CPU Inference**: If no CUDA GPU is detected, models will run on CPU, which is significantly slower.
* **LLaVA-Med**: By default, it is disabled due to its high RAM requirement (~14GB on CPU). It also inherently only accepts one image per inference call, so the lateral view is dropped when this model is enabled.
* **Uncertainty Labels**: By default, uncertain/possible findings are mapped to negative. This behavior is adjustable in `baseline.yaml` (`labeler.uncertainty_policy`).
