"""ComfyUI nodes for SemanticSAM with pure PyTorch dependencies."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from .sam_backend import SamModelConfig, SamModelRuntime
from .semantic_backend_hf import UniversalSegmenter, UniversalSegmenterConfig
from .semantic_labeling import assign_labels_to_sam_masks

SAM_MODELS: Dict[str, Dict[str, object]] = {
    "SAM ViT-H": {
        "checkpoint": "custom_nodes/ComfyUI_SemanticSAM/ckpt/sam_vit_h_4b8939.pth",
        "model_type": "vit_h",
    },
    "SAM ViT-L": {
        "checkpoint": "custom_nodes/ComfyUI_SemanticSAM/ckpt/sam_vit_l_0b3195.pth",
        "model_type": "vit_l",
    },
    "SAM ViT-B": {
        "checkpoint": "custom_nodes/ComfyUI_SemanticSAM/ckpt/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
    },
}

SEGMENTER_PRESETS: Dict[str, str] = {
    "OneFormer ADE20K (Swin-T)": "shi-labs/oneformer_ade20k_swin_tiny",
    "OneFormer COCO (Swin-L)": "shi-labs/oneformer_coco_swin_large",
}

_SEGMENTER_CACHE: Dict[str, UniversalSegmenter] = {}
_SAM_CACHE: Dict[Tuple[str, str], SamModelRuntime] = {}


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path))


def list_sam_model() -> List[str]:
    return list(SAM_MODELS.keys())


def _resolve_torch_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_segmenter(model_id: str) -> UniversalSegmenter:
    segmenter = _SEGMENTER_CACHE.get(model_id)
    if segmenter is None:
        config = UniversalSegmenterConfig(model_id=model_id)
        segmenter = UniversalSegmenter(config)
        _SEGMENTER_CACHE[model_id] = segmenter
    return segmenter


def _get_sam_runtime(model_name: str, device: torch.device) -> SamModelRuntime:
    cache_key = (model_name, str(device))
    runtime = _SAM_CACHE.get(cache_key)
    if runtime is not None:
        return runtime
    metadata = SAM_MODELS[model_name]
    checkpoint_path = _resolve_checkpoint_path(str(metadata["checkpoint"]))
    mask_generator_kwargs = dict(metadata.get("mask_generator", {}))
    config = SamModelConfig(
        checkpoint_path=checkpoint_path,
        model_type=str(metadata.get("model_type", "vit_h")),
        device=str(device),
        mask_generator_kwargs=mask_generator_kwargs,
    )
    runtime = SamModelRuntime(config)
    _SAM_CACHE[cache_key] = runtime
    return runtime


def sort_masks(masks: Iterable[np.ndarray], scores: np.ndarray, thresh: float) -> List[np.ndarray]:
    if scores.ndim:
        order = np.argsort(scores)[::-1]
    else:
        order = np.array([0], dtype=np.int64)
    sorted_masks: List[np.ndarray] = []
    for idx in order:
        iou = float(scores[idx])
        if iou < thresh:
            continue
        binary_mask = masks[idx].astype(np.uint8)
        sorted_masks.append(binary_mask)
    return sorted_masks


@dataclass
class SemanticSAMRuntime:
    sam: SamModelRuntime
    segmenter: UniversalSegmenter
    last_metadata: List[Dict[str, object]] = None

    def __post_init__(self) -> None:
        if self.last_metadata is None:
            self.last_metadata = []

    @property
    def device(self) -> torch.device:
        return self.sam.device

    def predict(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        return self.sam.predict_masks(
            image,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )


class PointPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_coord": ("INT", {}),
                "y_coord": ("INT", {}),
            },
            "optional": {},
        }

    CATEGORY = "SemanticSAM"
    RETURN_TYPES = ("POINTS",)
    FUNCTION = "main"

    def main(self, x_coord: int, y_coord: int):
        return ([[x_coord, y_coord]],)


class SemanticSAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(),),
            },
            "optional": {
                "segmenter": (
                    list(SEGMENTER_PRESETS.keys()),
                    {"default": "OneFormer ADE20K (Swin-T)"},
                ),
            },
        }

    CATEGORY = "SemanticSAM"
    RETURN_TYPES = ("SemanticSAM_Model",)
    FUNCTION = "main"

    def main(self, model_name: str, segmenter: str = "OneFormer ADE20K (Swin-T)"):
        device = _resolve_torch_device()
        sam_runtime = _get_sam_runtime(model_name, device)
        segmenter_id = SEGMENTER_PRESETS.get(segmenter, segmenter)
        runtime = SemanticSAMRuntime(sam_runtime, _get_segmenter(segmenter_id))
        return (runtime,)


class SemanticSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SemanticSAM_Model", {}),
                "image": ("IMAGE", {}),
                "points": ("POINTS", {}),
                "expand": ("INT", {"default": 0, "min": -10, "max": 10, "step": 1}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1, "step": 0.01},
                ),
                "num_masks": ("INT", {"default": 6, "min": 1, "max": 6, "step": 1}),
            },
            "optional": {},
        }

    CATEGORY = "SemanticSAM"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "main"

    def main(
        self,
        model: SemanticSAMRuntime,
        image: torch.Tensor,
        points: List[List[float]],
        expand: int,
        threshold: float = 0.5,
        num_masks: int = 0,
    ):
        batched_image = image[0]
        image_np = np.clip(batched_image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        pil_image = ToPILImage()(batched_image.permute(2, 0, 1))

        h, w = image_np.shape[:2]
        point_coords = np.array(points, dtype=np.float32)
        point_labels = None
        if point_coords.size:
            point_labels = np.ones(len(point_coords), dtype=np.int64)

        masks, scores = model.predict(
            image_np,
            point_coords=point_coords if point_coords.size else None,
            point_labels=point_labels,
            multimask_output=True,
        )

        if num_masks > 0:
            masks = masks[:num_masks]
            scores = scores[:num_masks]

        sorted_masks = sort_masks(masks, scores, threshold)
        if expand != 0 and sorted_masks:
            sorted_masks = expand_mask(np.array(sorted_masks), expand=expand)

        resized_masks = [
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            for mask in sorted_masks
        ]

        rgba_imgs, masks_tensor = masks2rgba(image[0], resized_masks)

        metadata: List[Dict[str, object]] = []
        if resized_masks:
            semantic_map, id2label = model.segmenter.predict_semantic(pil_image)
            metadata = assign_labels_to_sam_masks(semantic_map, id2label, resized_masks)
        model.last_metadata = metadata
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        return (rgba_imgs, masks_tensor, metadata_json)


NODE_CLASS_MAPPINGS = {
    "SemanticSAMLoader": SemanticSAMLoader,
    "SemanticSAMSegment": SemanticSAMSegment,
    "PointPrompt": PointPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticSAMLoader": "SemanticSAM Loader",
    "SemanticSAMSegment": "SemanticSAM Segment",
    "PointPrompt": "Point Prompt",
}


def expand_mask(mask: np.ndarray, expand: int = 0, tapered_corners: bool = True) -> np.ndarray:
    mask = mask.astype(np.float32)
    if expand == 0:
        return mask

    height = mask.shape[1]
    width = mask.shape[2]
    background = np.zeros((height + 2, width + 2), dtype=np.uint8)
    masks = []
    kernel = np.ones((5, 5), np.uint8)

    for m in mask:
        background[1:-1, 1:-1] = m
        border = cv2.dilate(background, kernel, iterations=abs(expand))
        mask_pad = border[1:-1, 1:-1]
        if expand < 0:
            mask_pad = cv2.erode(background, kernel, iterations=abs(expand))[1:-1, 1:-1]
        if tapered_corners:
            mask_pad = cv2.GaussianBlur(mask_pad, (3, 3), 0)
        masks.append(mask_pad)

    return np.array(masks)


def masks2rgba(image: torch.Tensor, masks: List[np.ndarray]):
    batch: List[torch.Tensor] = []
    mask_stack: List[torch.Tensor] = []
    image_np = image.detach().cpu().numpy()
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        color_mask = np.zeros_like(image_np)
        color_mask[:, :, 0] = mask_uint8
        color_mask[:, :, 1] = mask_uint8
        color_mask[:, :, 2] = mask_uint8
        alpha = mask_uint8.astype(np.float32)
        rgba = np.concatenate([color_mask, alpha[..., None]], axis=-1)
        batch.append(torch.from_numpy(rgba / 255.0).float())
        mask_stack.append(torch.from_numpy(mask).unsqueeze(0).float())
    if not batch:
        empty = torch.zeros((1, *image.shape[:2], 4), dtype=torch.float32)
        return empty, torch.zeros((1, 1, image.shape[0], image.shape[1]), dtype=torch.float32)
    return torch.stack(batch), torch.stack(mask_stack)
