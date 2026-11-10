"""
src/models/internvl2.py
-------------------------------------------------------------------------------
InternVL2 wrapper for chest X-ray report generation.

Checkpoint  : OpenGVLab/InternVL2-1B  or  OpenGVLab/InternVL2-2B
Architecture: InternViT-300M vision encoder + InternLM2 / Phi-3 Mini LLM
Multi-image : InternVL2 natively supports multi-image via special <image> tokens.
              We pass both PA and lateral views when available.
VRAM needed : ~2 GB (1B, float16) | ~4 GB (2B, float16)

Notes       : InternVL2 requires trust_remote_code=True.
              The conversation format uses a system prompt + user message
              with <image> tokens as placeholders for each image.

References:
  https://huggingface.co/OpenGVLab/InternVL2-1B
  https://huggingface.co/OpenGVLab/InternVL2-2B
  https://internvl.readthedocs.io/en/latest/
"""

from __future__ import annotations

import gc
import logging
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)


class InternVL2Model(BaseReportModel):
    """
    Wrapper around OpenGVLab/InternVL2-1B or InternVL2-2B.

    InternVL2 accepts multiple images natively through <image> token placeholders
    in the conversation. We pass all available X-ray views.
    """

    def __init__(self, checkpoint: str = "OpenGVLab/InternVL2-1B",
                 device: str = "cuda", dtype: str = "float16"):
        super().__init__(name="internvl2", checkpoint=checkpoint,
                         device=device, dtype=dtype)
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        logger.info(f"Loading InternVL2 from '{self.checkpoint}' ...")
        try:
            # InternVL2's remote code imports LlamaForCausalLM which checks
            # flash_attn availability via importlib.util.find_spec. Pre-register
            # a stub with a valid __spec__ so the check passes without crashing.
            import sys, types, importlib.util
            if "flash_attn" not in sys.modules:
                _mock = types.ModuleType("flash_attn")
                _mock.__spec__ = importlib.util.spec_from_loader(
                    "flash_attn", loader=None
                )
                _mock.__version__ = "0.0.0"
                sys.modules["flash_attn"] = _mock
                sys.modules["flash_attn.flash_attn_interface"] = _mock

            from transformers import AutoModel, AutoTokenizer
            import torch

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                use_fast=False,
            )
            self._model = AutoModel.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto",
            )
            self._model.eval()
            self._loaded = True
            logger.info("InternVL2 loaded successfully.")

        except ImportError:
            logger.error("transformers not installed.")
            raise
        except Exception as exc:
            logger.error(f"Failed to load InternVL2: {exc}")
            raise

    def _preprocess_image(self, pil_img: Image.Image):
        """Preprocess a single PIL image into InternVL2's pixel_values format."""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        INPUT_SIZE = 448

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((INPUT_SIZE, INPUT_SIZE),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(pil_img)

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

        # Build pixel_values: stack all images
        pixel_tensors = [self._preprocess_image(img) for img in images]
        pixel_values = torch.stack(pixel_tensors, dim=0).to(
            device=self._model.device,
            dtype=torch.float16 if self.dtype == "float16" else torch.float32,
        )

        # Build question with <image> placeholders for each view
        image_tokens = "\n".join(["<image>"] * len(images))
        question = f"{image_tokens}\n{prompt}"

        gen_cfg = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample and temperature > 0:
            gen_cfg["temperature"] = temperature

        try:
            # InternVL2 chat API
            response = self._model.chat(
                self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=gen_cfg,
            )
            return str(response).strip()
        except Exception as exc:
            logger.warning(f"InternVL2 chat() failed ({exc}); falling back to generate().")
            # Fallback: encode manually
            inputs = self._tokenizer(question, return_tensors="pt").to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, pixel_values=pixel_values, **gen_cfg)
            return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def unload(self) -> None:
        logger.info("Unloading InternVL2 ...")
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
        logger.info("InternVL2 unloaded.")
