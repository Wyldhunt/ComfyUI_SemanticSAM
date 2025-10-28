"""Minimal predictor for Semantic SAM without Detectron2 dependencies."""
from __future__ import annotations

from typing import Optional

import torch


class SemanticSAMPredictor:
    def __init__(self, model, thresh: float = 0.5):
        self.model = model
        self.thresh = thresh
        self.point: Optional[torch.Tensor] = None

    def predict(self, image_ori, image, point=None):
        width = image_ori.shape[1]
        height = image_ori.shape[0]

        data = {"image": image, "height": height, "width": width}
        if point is None:
            point = torch.tensor([[0.5, 0.5, 0.006, 0.006]], device=image.device)
        else:
            point = torch.tensor(point, device=image.device)
            point = torch.cat([point, point.new_tensor([[0.005, 0.005]])], dim=-1)

        self.point = point[:, :2].clone() * torch.tensor(
            [width, height], device=point.device
        )

        data["targets"] = [dict()]
        data["targets"][0]["points"] = point
        data["targets"][0]["pb"] = point.new_tensor([0.0])

        batch_inputs = [data]
        masks, ious = self.model.model.evaluate_demo(batch_inputs)
        return masks, ious
