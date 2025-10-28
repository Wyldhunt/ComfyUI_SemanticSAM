"""ComfyUI nodes for SemanticSAM with Hugging Face semantics."""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from .semantic_backend_hf import UniversalSegmenter, UniversalSegmenterConfig
from .semantic_labeling import assign_labels_to_sam_masks
from .semantic_sam import SemanticSAMPredictor, build_semantic_sam

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sam_model_list = {
    "L": {"model_url": "custom_nodes/ComfyUI_SemanticSAM/ckpt/swinl_only_sam_many2many.pth"},
    "T": {"model_url": "custom_nodes/ComfyUI_SemanticSAM/ckpt/swint_only_sam_many2many.pth"},
}

SEGMENTER_PRESETS: Dict[str, str] = {
    "OneFormer ADE20K (Swin-T)": "shi-labs/oneformer_ade20k_swin_tiny",
    "OneFormer COCO (Swin-L)": "shi-labs/oneformer_coco_swin_large",
}

_SEGMENTER_CACHE: Dict[str, UniversalSegmenter] = {}


def list_sam_model():
    return list(sam_model_list.keys())


def _resolve_torch_device(preferred: Optional[str] = None) -> torch.device:
    """Return a usable torch.device for SemanticSAM inference."""

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


def sort_masks(masks, ious, thresh):
    ious = ious[0, 0]
    ids = torch.argsort(ious, descending=True)
    sorted_masks = []
    for mask, iou in zip(masks[ids], ious[ids]):
        iou = float(iou)
        if iou < thresh:
            continue
        binary_mask = (mask.cpu().detach().numpy() > 0).astype(np.uint8)
        sorted_masks.append(binary_mask)
    return sorted_masks


class SemanticSAMRuntime:
    def __init__(self, predictor: SemanticSAMPredictor, segmenter: UniversalSegmenter):
        self.predictor = predictor
        self.segmenter = segmenter
        self.model = predictor.model
        self.last_metadata: List[Dict[str, object]] = []

    def predict(self, *args, **kwargs):
        with torch.inference_mode():
            return self.predictor.predict(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


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

    def main(self, x_coord, y_coord):
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

    def main(self, model_name, segmenter="OneFormer ADE20K (Swin-T)"):
        ckpt_path = sam_model_list[model_name]["model_url"]
        device = _resolve_torch_device()
        predictor = SemanticSAMPredictor(
            build_semantic_sam(model_type=model_name, ckpt=ckpt_path, device=device)
        )
        segmenter_id = SEGMENTER_PRESETS.get(segmenter, segmenter)
        runtime = SemanticSAMRuntime(predictor, _get_segmenter(segmenter_id))
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
        image,
        points: List[List[float]],
        expand: int,
        threshold: float = 0.3,
        num_masks: int = 0,
    ):
        tensor_image = image[0].permute(2, 0, 1).detach().cpu()
        pil_image = ToPILImage()(tensor_image)
        original_image, input_image = self.prepare_image(tensor_image, model.device)

        h = image.shape[1]
        w = image.shape[2]
        points_np = np.array(points, dtype=np.float32)
        point_input = None
        if points_np.size:
            points_np[:, 0] = points_np[:, 0] / w
            points_np[:, 1] = points_np[:, 1] / h
            point_input = points_np
        masks, ious = model.predict(original_image, input_image, point=point_input)

        if num_masks > 0:
            masks = masks[0:num_masks]
        sorted_masks = sort_masks(masks, ious, threshold)
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

    def prepare_image(self, tensor_image: torch.Tensor, device: torch.device):
        t = [ToPILImage(), transforms.Resize(640, interpolation=Image.BICUBIC)]
        transform1 = transforms.Compose(t)
        image_ori = transform1(tensor_image)
        image_np = np.asarray(image_ori)
        images = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(device)
        return image_np, images


def expand_mask(mask, expand=0, tapered_corners=True):
    mask = np.array(mask.astype(np.uint8))
    c = 0 if tapered_corners else 1
    kernel = np.array(
        [
            [c, 1, c],
            [1, 1, 1],
            [c, 1, c],
        ]
    )
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        out.append(np.array(output))
    return out


def masks2rgba(image: torch.Tensor, masks: List[np.ndarray]):
    rgba_imgs = []
    masks_tensor = []
    base_image = image.detach().cpu()
    height, width = base_image.shape[0], base_image.shape[1]
    for mask in masks:
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.shape != (height, width):
            mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        rgba_tensor = mask2rgba(base_image, mask_arr)
        rgba_imgs.append(rgba_tensor)
        masks_tensor.append(torch.from_numpy(mask_arr).unsqueeze(dim=0))
    if masks_tensor:
        masks_tensor = torch.cat(masks_tensor, dim=0)
        rgba_imgs = torch.cat(rgba_imgs, dim=0)
    else:
        masks_tensor = torch.zeros((0, height, width), dtype=torch.float32)
        rgba_imgs = torch.zeros((0, height, width, 4), dtype=torch.float32)
    return rgba_imgs, masks_tensor


def mask2rgba(image: torch.Tensor, mask: np.ndarray):
    rgba_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    rgba_image[:, :, :3] = image.detach().cpu().numpy().astype(np.float32)
    rgba_image[:, :, 3] = mask
    rgba_tensor = torch.from_numpy(rgba_image).unsqueeze(dim=0)
    return rgba_tensor


NODE_CLASS_MAPPINGS = {
    "SemanticSAMLoader": SemanticSAMLoader,
    "SemanticSAMSegment": SemanticSAMSegment,
    "PointPrompt": PointPrompt,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticSAMLoader": "SemanticSAM Loader",
    "SemanticSAMSegment": "SemanticSAM Segment",
    "PointPrompt": "SemanticSAM Point Prompt",
}
