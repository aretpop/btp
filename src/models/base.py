"""
src/models/base.py
-------------------------------------------------------------------------------
Abstract base class for all report-generation model wrappers.

Every model must implement:
  - load()          : load weights / tokenizer into memory
  - generate()      : produce a report string from a list of PIL Images + prompt
  - unload()        : free GPU/CPU memory (called after inference is done)

The evaluation pipeline calls models exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from PIL import Image


class BaseReportModel(ABC):
    """
    Abstract base class for a chest X-ray report generation model.

    Subclasses must implement load(), generate(), and unload().
    """

    def __init__(self, name: str, checkpoint: str, device: str = "cpu",
                 dtype: str = "float32"):
        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""
        ...

    @abstractmethod
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

        Parameters
        ----------
        images         : list of PIL Image objects for this examination
        prompt         : the clinical instruction string
        max_new_tokens : generation length cap
        temperature    : sampling temperature (0.0 = greedy)
        do_sample      : whether to sample or use greedy decoding

        Returns
        -------
        str : the generated report text
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, checkpoint={self.checkpoint!r})"

    def is_loaded(self) -> bool:
        return self._loaded
