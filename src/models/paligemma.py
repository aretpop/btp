"""
src/models/paligemma.py
-------------------------------------------------------------------------------
PaliGemma 2 wrapper for chest X-ray report generation.

Checkpoint  : google/paligemma2-3b-pt-224  (or -448 for higher resolution)
Architecture: SigLIP vision encoder + Gemma 2 LLM
Multi-image : Single-image per call; we use the PA (frontal) view as primary.
              If only one image is available that is fine too.
VRAM needed : ~6 GB (float16, 3B) — fits on RTX 4060 8 GB
Notes       : PaliGemma is a prefix-LM; the prompt is prepended to the image
              tokens. Pass the prompt as plain text, no chat template needed.

References:
  https://huggingface.co/google/paligemma2-3b-pt-224
  https://huggingface.co/docs/transformers/model_doc/paligemma
"""

from __future__ import annotations

import gc
import logging
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)


class PaliGemma2Model(BaseReportModel):
    """
    Wrapper around google/paligemma2-3b-pt-224 (or -448).

    PaliGemma is a prefix language model: the image is tokenised and prepended
    to the text prompt, then the model auto-regressively generates the suffix.
    The -pt- checkpoints are base pretrained; the -ft- variants are task-tuned.

    Multi-image limitation:
      PaliGemma 2 (like its predecessor) accepts exactly one image per call.
      For IU X-Ray dual-image exams we use only image[0] (PA/frontal view).
    """

    def __init__(self, checkpoint: str = "google/paligemma2-3b-pt-224",
                 device: str = "cuda", dtype: str = "float16"):
        super().__init__(name="paligemma2", checkpoint=checkpoint,
                         device=device, dtype=dtype)
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading PaliGemma2 from '{self.checkpoint}' ...")
        try:
            from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
            import torch

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            self._processor = AutoProcessor.from_pretrained(self.checkpoint)
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            self._model.eval()
            self._loaded = True
            logger.info("PaliGemma2 loaded successfully.")

        except ImportError:
            logger.error("transformers not installed.")
            raise
        except Exception as exc:
            logger.error(f"Failed to load PaliGemma2: {exc}")
            raise

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        if len(images) > 1:
            logger.debug(
                f"PaliGemma2: received {len(images)} images; using only image[0] (PA view)."
            )
        primary_image = images[0].convert("RGB")

        inputs = self._processor(
            images=primary_image,
            text=prompt,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[:, input_len:]
        generated = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        return generated.strip()

    def unload(self) -> None:
        logger.info("Unloading PaliGemma2 ...")
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("PaliGemma2 unloaded.")
