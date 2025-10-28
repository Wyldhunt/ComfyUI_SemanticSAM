"""Utilities for mapping SAM masks to semantic labels."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _compute_bounding_box(mask: np.ndarray) -> List[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def assign_labels_to_sam_masks(
    semantic_ids: np.ndarray,
    id2label: Dict[int, str],
    sam_masks: Iterable[np.ndarray],
) -> List[Dict[str, object]]:
    """Assign semantic labels to SAM masks via majority vote."""

    results: List[Dict[str, object]] = []
    for index, raw_mask in enumerate(sam_masks):
        mask = np.asarray(raw_mask).astype(bool)
        mask_pixels = semantic_ids[mask]
        area = int(mask.sum())
        if mask_pixels.size == 0 or area == 0:
            results.append(
                {
                    "mask_index": index,
                    "label_id": None,
                    "label": None,
                    "confidence": 0.0,
                    "area": 0,
                    "bbox": [0, 0, 0, 0],
                }
            )
            continue

        labels, counts = np.unique(mask_pixels, return_counts=True)
        majority_idx = int(np.argmax(counts))
        label_id = int(labels[majority_idx])
        label = id2label.get(label_id, str(label_id))
        confidence = float(counts[majority_idx] / area)
        bbox = _compute_bounding_box(mask)

        results.append(
            {
                "mask_index": index,
                "label_id": label_id,
                "label": label,
                "confidence": confidence,
                "area": area,
                "bbox": bbox,
            }
        )

    return results
