"""
src/models/llava_med.py
-------------------------------------------------------------------------------
LLaVA-Med wrapper for chest X-ray report generation.

Checkpoint  : microsoft/llava-med-v1.5-mistral-7b
Architecture: LLaVA-1.5 with Mistral-7B backbone, medically fine-tuned
Multi-image : LLaVA architecture accepts one image per call.
              Strategy: use only image 0 (PA view) as primary; log limitation.
VRAM needed : ~14 GB (float32 CPU) | ~7 GB (float16 GPU)
Status      : DISABLED by default in baseline.yaml due to RAM constraints.
              Set models[2].enabled: true to attempt.

References:
  https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b
  https://github.com/microsoft/LLaVA-Med
"""

from __future__ import annotations

import gc
import logging
from typing import List

from PIL import Image

from .base import BaseReportModel

logger = logging.getLogger(__name__)

# Known limitation - documented here explicitly
MULTI_IMAGE_LIMITATION = (
    "LLaVA-Med (LLaVA-1.5 backbone) accepts exactly one image per call. "
    "For dual-image IU X-Ray exams, only the first image (PA view, index 0) "
    "is passed to the model. The lateral view is not used. "
    "This is a documented limitation of this model for this evaluation."
)


class LLaVAMedModel(BaseReportModel):
    """
    Wrapper around microsoft/llava-med-v1.5-mistral-7b.

    IMPORTANT: This model requires ~14 GB RAM on CPU or ~7 GB VRAM on GPU.
    It is disabled by default in baseline.yaml. Enable only if your system
    has sufficient resources.

    Multi-image limitation:
      LLaVA architecture (both 1.5 and Med variants) supports exactly one
      image per inference call. For the dual-image IU X-Ray records, we
      use only the first image (typically the PA/frontal view).
      This limitation is logged for every sample.
    """

    def __init__(self, checkpoint: str = "microsoft/llava-med-v1.5-mistral-7b",
                 device: str = "cpu", dtype: str = "float32"):
        super().__init__(
            name="llava-med",
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )
        self._model = None
        self._processor = None
        logger.warning(MULTI_IMAGE_LIMITATION)

    def load(self) -> None:
        logger.info(f"Loading LLaVA-Med from '{self.checkpoint}' ...")
        logger.warning(
            f"LLaVA-Med requires ~14 GB RAM (CPU float32) or ~7 GB VRAM (GPU float16). "
            f"Current device: {self.device}."
        )
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoConfig, LlavaConfig, LlavaProcessor
            import torch

            class LlavaMistralConfig(LlavaConfig):
                model_type = "llava_mistral"
            try:
                AutoConfig.register("llava_mistral", LlavaMistralConfig)
            except Exception:
                pass

            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self._processor = LlavaProcessor.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
            )
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
            self._model.eval()
            self._loaded = True
            logger.info("LLaVA-Med loaded successfully.")

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise
        except Exception as exc:
            logger.error(f"Failed to load LLaVA-Med: {exc}")
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
        Generate a radiology report.

        Only the first image is used (LLaVA-1.5 limitation).
        The limitation is logged but does not suppress output.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        if len(images) > 1:
            logger.debug(
                f"LLaVA-Med: received {len(images)} images; using only image[0] "
                f"(PA view). Lateral view dropped. See MULTI_IMAGE_LIMITATION."
            )
        primary_image = images[0]

        # LLaVA-1.5 conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        except Exception:
            # Fallback prompt format for older processor versions
            text = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self._processor(
            text=text,
            images=primary_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        generated = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        return generated.strip()

    def unload(self) -> None:
        logger.info("Unloading LLaVA-Med ...")
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
        logger.info("LLaVA-Med unloaded.")
