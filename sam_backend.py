"""Segment Anything backend helpers without Detectron dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


@dataclass
class SamModelConfig:
    """Configuration object for loading a Segment Anything model."""

    checkpoint_path: str
    model_type: str = "vit_h"
    device: Optional[str] = None
    mask_generator_kwargs: Dict[str, object] = field(default_factory=dict)


class SamModelRuntime:
    """Wrapper providing both automatic and prompt-based SAM inference."""

    def __init__(self, config: SamModelConfig) -> None:
        self.config = config
        self._device = self._resolve_device(config.device)
        self._model = self._load_model()
        self._automatic_generator = SamAutomaticMaskGenerator(
            self._model,
            **config.mask_generator_kwargs,
        )
        self._predictor = SamPredictor(self._model)

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_model(self) -> torch.nn.Module:
        if self.config.model_type not in sam_model_registry:
            available = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(
                f"Unknown SAM model type '{self.config.model_type}'. Available: {available}."
            )
        model = sam_model_registry[self.config.model_type](
            checkpoint=self.config.checkpoint_path
        )
        model.to(self._device)
        model.eval()
        return model

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict_masks(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate masks for ``image`` using SAM."""

        if point_coords is None or point_coords.size == 0:
            return self._generate_automatic(image)
        return self._generate_with_points(
            image,
            point_coords,
            point_labels,
            multimask_output=multimask_output,
        )

    def _generate_automatic(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        mask_data = self._automatic_generator.generate(image)
        masks = [entry["segmentation"].astype(np.uint8) for entry in mask_data]
        scores = np.array(
            [float(entry.get("predicted_iou", 0.0)) for entry in mask_data],
            dtype=np.float32,
        )
        return masks, scores

    def _generate_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: Optional[np.ndarray] = None,
        *,
        multimask_output: bool,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        self._predictor.set_image(image)
        if point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=np.int64)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        return [mask.astype(np.uint8) for mask in masks], scores.astype(np.float32)

    def generate_metadata(self, masks: Iterable[np.ndarray]) -> List[Dict[str, float]]:
        metadata = []
        for mask in masks:
            area = float(mask.sum())
            metadata.append({"area": area})
        return metadata
