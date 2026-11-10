"""
src/dataset.py
-------------------------------------------------------------------------------
IU X-Ray dataset loader and validator.

Responsibilities
----------------
* Parse JSONL split files (train / val / test).
* Resolve every /iu_xray/image/... path to an absolute Windows path.
* Validate the presence of every referenced image file.
* Report missing / malformed records without silently discarding them.
* Expose a clean list of validated samples for downstream use.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CXRSample:
    """A single IU X-Ray examination record."""
    sample_id: str                  # e.g. "CXR3030_IM-1405"
    image_paths: List[Path]         # resolved absolute paths to .png files
    reference_report: str           # ground-truth radiology report
    raw_image_refs: List[str]       # original strings from the JSONL
    query: str                      # original prompt/query string


@dataclass
class ValidationReport:
    """Summary of dataset validation results."""
    total_records: int = 0
    valid: int = 0
    invalid: int = 0
    missing_response: List[int] = field(default_factory=list)
    missing_images: List[Tuple[int, str]] = field(default_factory=list)
    empty_response: List[int] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Dataset Validation Summary")
        logger.info("=" * 60)
        logger.info(f"  Total records     : {self.total_records}")
        logger.info(f"  Valid             : {self.valid}")
        logger.info(f"  Invalid           : {self.invalid}")
        if self.missing_response:
            logger.warning(f"  Missing 'response': {len(self.missing_response)} records (lines {self.missing_response[:10]}...)")
        if self.empty_response:
            logger.warning(f"  Empty 'response'  : {len(self.empty_response)} records (lines {self.empty_response[:10]}...)")
        if self.missing_images:
            logger.warning(f"  Missing images    : {len(self.missing_images)}")
            for line_no, path in self.missing_images[:10]:
                logger.warning(f"    line {line_no}: {path}")
        logger.info("=" * 60)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("Dataset Validation Summary")
        print("=" * 60)
        print(f"  Total records     : {self.total_records}")
        print(f"  Valid             : {self.valid}")
        print(f"  Invalid           : {self.invalid}")
        if self.missing_response:
            print(f"  Missing 'response': {len(self.missing_response)} records")
            print(f"    First few lines : {self.missing_response[:5]}")
        if self.empty_response:
            print(f"  Empty 'response'  : {len(self.empty_response)} records")
            print(f"    First few lines : {self.empty_response[:5]}")
        if self.missing_images:
            print(f"  Missing images    : {len(self.missing_images)}")
            for line_no, path in self.missing_images[:5]:
                print(f"    line {line_no}: {path}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_image_path(
    raw_ref: str,
    dataset_root: Path,
    image_root: Path,
    jsonl_prefix: str,
) -> Path:
    """
    Convert a JSONL image reference such as

        /iu_xray/image/CXR3030_IM-1405/0.png

    into an absolute local path by:
      1. Stripping the known JSONL prefix  ``/iu_xray/image/``
      2. Joining the remainder with  ``image_root``

    Parameters
    ----------
    raw_ref       : e.g. "/iu_xray/image/CXR3030_IM-1405/0.png"
    dataset_root  : absolute Path to the IU-Xray folder
    image_root    : absolute Path to the folder containing CXR* subdirectories
    jsonl_prefix  : the prefix to strip, e.g. "/iu_xray/image/"
    """
    if raw_ref.startswith(jsonl_prefix):
        relative_part = raw_ref[len(jsonl_prefix):]
    else:
        # Fallback: treat the whole thing as relative
        relative_part = raw_ref.lstrip("/")
        logger.debug(f"Unexpected image prefix in: {raw_ref!r}; treating as relative.")

    return image_root / relative_part


def _extract_sample_id(image_refs: List[str]) -> str:
    """
    Derive a human-readable sample ID from the first image reference.
    E.g. "/iu_xray/image/CXR3030_IM-1405/0.png" -> "CXR3030_IM-1405"
    """
    first = image_refs[0] if image_refs else "UNKNOWN"
    # Take the second-to-last path component (the CXR folder name)
    parts = first.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return Path(first).stem


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

class IUXRayDataset:
    """
    Loads and validates one split of the IU X-Ray dataset.

    Usage
    -----
    >>> ds = IUXRayDataset.from_config(cfg, split="test")
    >>> sample = ds.samples[0]
    >>> print(sample.reference_report)
    """

    def __init__(
        self,
        samples: List[CXRSample],
        validation_report: ValidationReport,
        split: str,
    ) -> None:
        self.samples = samples
        self.validation_report = validation_report
        self.split = split

    # ── constructor ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        split: str = "test",
        max_samples: Optional[int] = None,
        require_all_images: bool = True,
    ) -> "IUXRayDataset":
        """
        Build a dataset from a loaded YAML config dict.

        Parameters
        ----------
        cfg              : parsed config dict (from baseline.yaml)
        split            : "train" | "val" | "test"
        max_samples      : if set, truncate to this many records (smoke test)
        require_all_images: if True, records with any missing image are invalid
        """
        dataset_cfg = cfg["dataset"]
        dataset_root = Path(dataset_cfg["root"]).resolve()
        image_root = dataset_root / dataset_cfg["image_root_relative"]
        jsonl_prefix = dataset_cfg["jsonl_image_prefix"]

        split_file = dataset_root / dataset_cfg["splits"][split]

        return cls._load_split(
            split_file=split_file,
            dataset_root=dataset_root,
            image_root=image_root,
            jsonl_prefix=jsonl_prefix,
            split=split,
            max_samples=max_samples,
            require_all_images=require_all_images,
        )

    @classmethod
    def _load_split(
        cls,
        split_file: Path,
        dataset_root: Path,
        image_root: Path,
        jsonl_prefix: str,
        split: str,
        max_samples: Optional[int],
        require_all_images: bool,
    ) -> "IUXRayDataset":
        logger.info(f"Loading split '{split}' from: {split_file}")

        if not split_file.exists():
            raise FileNotFoundError(f"JSONL split file not found: {split_file}")

        vr = ValidationReport()
        samples: List[CXRSample] = []

        with open(split_file, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                vr.total_records += 1

                # ── Parse JSON ───────────────────────────────────────────────
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.error(f"  Line {line_no}: JSON parse error — {exc}")
                    vr.invalid += 1
                    continue

                # ── Validate 'response' ──────────────────────────────────────
                response = record.get("response", None)
                if response is None:
                    logger.warning(f"  Line {line_no}: missing 'response' field.")
                    vr.missing_response.append(line_no)
                    vr.invalid += 1
                    continue

                if not str(response).strip():
                    logger.warning(f"  Line {line_no}: empty 'response' field.")
                    vr.empty_response.append(line_no)
                    vr.invalid += 1
                    continue

                # ── Validate 'images' ────────────────────────────────────────
                raw_images: List[str] = record.get("images", [])
                if not raw_images:
                    logger.warning(f"  Line {line_no}: no images listed.")
                    vr.missing_images.append((line_no, "<no images field>"))
                    vr.invalid += 1
                    continue

                resolved_paths: List[Path] = []
                record_has_missing = False

                for raw_ref in raw_images:
                    abs_path = _resolve_image_path(
                        raw_ref, dataset_root, image_root, jsonl_prefix
                    )
                    if not abs_path.exists():
                        logger.warning(
                            f"  Line {line_no}: image not found: {abs_path}"
                        )
                        vr.missing_images.append((line_no, str(abs_path)))
                        record_has_missing = True
                    else:
                        resolved_paths.append(abs_path)

                if require_all_images and record_has_missing:
                    vr.invalid += 1
                    continue

                if not resolved_paths:
                    vr.invalid += 1
                    continue

                # ── Build sample ─────────────────────────────────────────────
                sample_id = _extract_sample_id(raw_images)
                sample = CXRSample(
                    sample_id=sample_id,
                    image_paths=resolved_paths,
                    reference_report=str(response).strip(),
                    raw_image_refs=raw_images,
                    query=record.get("query", ""),
                )
                samples.append(sample)
                vr.valid += 1

                if max_samples and len(samples) >= max_samples:
                    logger.info(
                        f"  Reached max_samples={max_samples}; stopping early."
                    )
                    break

        vr.invalid = vr.total_records - vr.valid
        logger.info(
            f"Loaded {vr.valid}/{vr.total_records} valid records from '{split}'."
        )

        return cls(samples=samples, validation_report=vr, split=split)

    # ── convenience ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CXRSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
