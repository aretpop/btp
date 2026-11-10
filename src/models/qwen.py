"""
src/models/qwen.py
-------------------------------------------------------------------------------
Qwen2-VL wrapper for chest X-ray report generation.

Checkpoint : Qwen/Qwen2-VL-2B-Instruct
Architecture: 2B vision-language model with native multi-image support
Multi-image : Fully supported via chat template with multiple <image> tokens
VRAM needed : ~4 GB (float32) | ~2.5 GB (float16 GPU)

References:
  https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  https://github.com/QwenLM/Qwen2-VL
"""

from __future__ import annotations

import gc
import logging
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)


class Qwen2VLModel(BaseReportModel):
    """
    Wrapper around the Qwen2-VL-2B-Instruct vision-language model.

    Qwen2-VL natively supports multiple images in a single conversation turn
    using the messages format:
        [{"role": "user", "content": [
            {"type": "image", "image": <PIL Image>},
            {"type": "image", "image": <PIL Image>},
            {"type": "text",  "text":  <prompt>},
        ]}]

    Both images from the IU X-Ray examination are passed together.
    """

    def __init__(self, checkpoint: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cpu", dtype: str = "float32"):
        super().__init__(
            name="qwen2-vl",
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading Qwen2-VL from '{self.checkpoint}' ...")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("Qwen2-VL loaded successfully.")

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise
        except Exception as exc:
            logger.error(f"Failed to load Qwen2-VL: {exc}")
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
        Generate a radiology report from one or more X-ray images.

        All images are passed together in a single message, allowing the model
        to reason jointly over the PA and lateral views.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Build multi-modal message content
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            from qwen_vl_utils import process_vision_info
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            # Fallback: use processor directly without qwen_vl_utils
            logger.debug("qwen_vl_utils not available; using processor fallback.")
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        generated = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        return generated.strip()

    def unload(self) -> None:
        logger.info("Unloading Qwen2-VL ...")
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
        logger.info("Qwen2-VL unloaded.")
