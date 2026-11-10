"""
src/models/moondream.py
-------------------------------------------------------------------------------
Moondream2 wrapper for chest X-ray report generation.

Checkpoint : vikhyatk/moondream2  (revision: 2025-01-09)
Architecture: ~2B vision-language model
Multi-image : Supported via concatenation / multiple image tokens
VRAM needed : ~4 GB (float32 CPU-only) | ~2 GB (float16 GPU)

References:
  https://github.com/vikhyatk/moondream
  https://huggingface.co/vikhyatk/moondream2
"""

from __future__ import annotations

import gc
import logging
import time
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)


class Moondream2Model(BaseReportModel):
    """
    Wrapper around the Moondream2 vision-language model.

    Moondream2 exposes a simple Python API:
        model.query(image, question)["answer"]
    For multi-image inputs, we pass each image individually and concatenate
    the resulting descriptions, which is the recommended approach for models
    that technically accept one image per call but the task has two views.

    NOTE: The 2025-01-09 revision of moondream2 added a batch_answer() API
    that can handle multiple images. We use the single-image query() path as
    a safe, compatible fallback.
    """

    def __init__(self, checkpoint: str = "vikhyatk/moondream2",
                 revision: str = "2025-01-09",
                 device: str = "cpu", dtype: str = "float32"):
        super().__init__(
            name="moondream2",
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        self.revision = revision
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        logger.info(f"Loading Moondream2 from '{self.checkpoint}' (revision={self.revision}) ...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("Moondream2 loaded successfully.")

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise
        except Exception as exc:
            logger.error(f"Failed to load Moondream2: {exc}")
            raise

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate a radiology report for one or more X-ray images.

        Multi-image strategy:
          We query each image independently and concatenate the results.
          This preserves all image information without dropping any view.
          The combined output is prefixed with view labels for clarity.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        reports = []
        with torch.inference_mode():
            for i, pil_img in enumerate(images):
                view_label = f"View {i} (PA)" if i == 0 else f"View {i} (lateral)"
                try:
                    enc_img = self._model.encode_image(pil_img)
                    if hasattr(self._model, "answer_question"):
                        # Older revision API
                        answer = self._model.answer_question(enc_img, prompt, self._tokenizer)
                    else:
                        # Newer revision API
                        answer = self._model.query(enc_img, prompt)["answer"]
                    reports.append(f"[{view_label}]: {answer}")
                except Exception as exc:
                    logger.warning(f"Failed to process image {i}: {exc}")
                    reports.append(f"[{view_label}]: <generation failed>")

        if len(reports) == 1:
            # Single image: strip the view label prefix for cleaner output
            result = reports[0].split("]: ", 1)[-1]
        else:
            result = " ".join(reports)

        return result.strip()

    def unload(self) -> None:
        """Free model from memory."""
        logger.info("Unloading Moondream2 ...")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Moondream2 unloaded.")
