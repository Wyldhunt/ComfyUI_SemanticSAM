"""Utilities for semantic segmentation using Hugging Face Transformers models."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoProcessor


@dataclass
class UniversalSegmenterConfig:
    """Configuration for :class:`UniversalSegmenter`."""

    model_id: str = "shi-labs/oneformer_ade20k_swin_tiny"
    task: str = "semantic"
    device: Optional[str] = None


class UniversalSegmenter:
    """A light-weight wrapper around Hugging Face segmentation models.

    The segmenter loads a "universal" segmentation model such as OneFormer or
    Mask2Former from the Transformers library and exposes a helper for obtaining
    semantic class ids for an input image. Only PyTorch kernels are required –
    no Detectron2 or custom CUDA operators are needed.
    """

    def __init__(self, config: Optional[UniversalSegmenterConfig] = None):
        self.config = config or UniversalSegmenterConfig()
        self._device = self._resolve_device(self.config.device)
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForUniversalSegmentation.from_pretrained(
            self.config.model_id
        )
        self._model.to(self._device)
        self._model.eval()
        # Normalise the label mapping to have integer keys.
        self._id2label: Dict[int, str] = {
            int(k): v for k, v in self._model.config.id2label.items()
        }

        # Cache whether the processor expects a task prompt (OneFormer) or not
        # (Mask2Former and friends).
        processor_signature = inspect.signature(self._processor.__call__)
        self._supports_task_inputs = "task_inputs" in processor_signature.parameters

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    def predict_semantic(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Return a semantic segmentation map for ``image``.

        Parameters
        ----------
        image:
            A PIL image in RGB format.

        Returns
        -------
        np.ndarray
            Semantic class ids with shape ``(H, W)`` and ``np.int32`` dtype.
        dict[int, str]
            Mapping from class id to a human readable label.
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        processor_kwargs = {"images": image, "return_tensors": "pt"}
        if self._supports_task_inputs:
            processor_kwargs["task_inputs"] = [self.config.task]
        inputs = self._processor(**processor_kwargs)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)

        target_size = torch.tensor([image.size[::-1]])  # (height, width)
        processed = self._processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_size,
        )[0]

        semantic_map = processed.to(torch.int32).cpu().numpy()
        return semantic_map, self._id2label


def batch_predict_semantic(
    segmenter: UniversalSegmenter, images: Sequence[Image.Image]
) -> Tuple[Sequence[np.ndarray], Dict[int, str]]:
    """Run :meth:`UniversalSegmenter.predict_semantic` on many images."""

    semantic_maps = []
    id2label = segmenter.id2label
    for image in images:
        semantic_map, _ = segmenter.predict_semantic(image)
        semantic_maps.append(semantic_map)
    return semantic_maps, id2label
