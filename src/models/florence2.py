"""
src/models/florence2.py
-------------------------------------------------------------------------------
Florence-2 wrapper for chest X-ray report generation.

Checkpoint  : microsoft/Florence-2-base  (0.23B) or Florence-2-large (0.77B)
Architecture: DaViT vision encoder + transformer encoder-decoder
Multi-image : Single image per call. We use image[0] (PA view).
VRAM needed : < 1.5 GB (base, float16) — extremely lightweight
Notes       : Florence-2 uses task-token prefixes like <DETAILED_CAPTION>.
              We use <DETAILED_CAPTION> + our clinical prompt as the task input.
              trust_remote_code=True is required.

References:
  https://huggingface.co/microsoft/Florence-2-base
  https://huggingface.co/microsoft/Florence-2-large
"""

from __future__ import annotations

import gc
import logging
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)

# Florence-2 task token for detailed image captioning
FLORENCE_TASK = "<DETAILED_CAPTION>"


class Florence2Model(BaseReportModel):
    """
    Wrapper around microsoft/Florence-2-base or Florence-2-large.

    Florence-2 is an encoder-decoder VLM that accepts task prompts.
    We use the <DETAILED_CAPTION> task and append the clinical instruction
    so the model focuses on clinically relevant findings.

    Multi-image limitation:
      Florence-2 accepts exactly one image per call.
      For IU X-Ray dual-image records, only image[0] (PA view) is used.
    """

    def __init__(self, checkpoint: str = "microsoft/Florence-2-base",
                 device: str = "cuda", dtype: str = "float16"):
        super().__init__(name="florence2", checkpoint=checkpoint,
                         device=device, dtype=dtype)
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading Florence-2 from '{self.checkpoint}' ...")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=True
            )
            # Florence-2's remote code calls is_flash_attn_2_available() which
            # uses importlib.util.find_spec("flash_attn"). We pre-register a stub
            # module with a valid __spec__ so find_spec() returns it gracefully
            # (returns None for version check) without crashing.
            import sys, types, importlib.util
            if "flash_attn" not in sys.modules:
                _mock = types.ModuleType("flash_attn")
                _mock.__spec__ = importlib.util.spec_from_loader(
                    "flash_attn", loader=None
                )
                _mock.__version__ = "0.0.0"
                sys.modules["flash_attn"] = _mock
                sys.modules["flash_attn.flash_attn_interface"] = _mock

            self._model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="sdpa",
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("Florence-2 loaded successfully.")

        except ImportError:
            logger.error("transformers not installed.")
            raise
        except Exception as exc:
            logger.error(f"Failed to load Florence-2: {exc}")
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
                f"Florence-2: received {len(images)} images; using only image[0] (PA view)."
            )
        primary_image = images[0].convert("RGB")

        # Florence-2 requires the task token to be the *only* text token.
        # We use <DETAILED_CAPTION> for maximum visual detail, then the
        # post-processor strips the task token from the output automatically.
        task_prompt = FLORENCE_TASK

        inputs = self._processor(
            text=task_prompt,
            images=primary_image,
            return_tensors="pt",
        ).to(self.device)

        # Cast pixel_values to model's dtype (float16) — the processor always
        # returns float32 regardless of the model's dtype, causing a type mismatch.
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            early_stopping=False,
            num_beams=3,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Florence-2 post-processes outputs via the processor
        generated_text = self._processor.batch_decode(
            output_ids, skip_special_tokens=False
        )[0]

        # Use the processor's post_process to clean the output
        try:
            parsed = self._processor.post_process_generation(
                generated_text,
                task=FLORENCE_TASK,
                image_size=(primary_image.width, primary_image.height),
            )
            result = parsed.get(FLORENCE_TASK, generated_text)
        except Exception:
            result = generated_text

        return str(result).strip()

    def unload(self) -> None:
        logger.info("Unloading Florence-2 ...")
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
        logger.info("Florence-2 unloaded.")
