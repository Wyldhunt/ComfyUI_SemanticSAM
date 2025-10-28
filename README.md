# ComfyUI SemanticSAM (modernised)

This project provides ComfyUI nodes for [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) masks combined with modern
Hugging Face semantic segmentation models. The node continues to expose the familiar loader/segment workflow while removing the
Detectron2/Mask2Former build requirements.

## What's new

* ✅ Pure PyTorch inference path – no Detectron2, no custom CUDA builds.
* ✅ Semantic labels are produced for every SAM mask using Hugging Face's OneFormer models.
* ✅ Metadata about every mask (label id, name, confidence, area, bounding box) is returned as a JSON string for easy storage in
  downstream systems.

## Installation

1. Clone the repository into your ComfyUI `custom_nodes` directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/eastoc/ComfyUI_SemanticSAM
   cd ComfyUI_SemanticSAM
   ```

2. Download the Semantic-SAM checkpoints into `custom_nodes/ComfyUI_SemanticSAM/ckpt` (same links as the upstream project):

   | Name | Backbone | Download |
   | --- | --- | --- |
   | Semantic-SAM SwinT | SwinT | [model](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swint_only_sam_many2many.pth) |
   | Semantic-SAM SwinL | SwinL | [model](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth) |

3. Install PyTorch (and torchvision) according to the official instructions for your platform.

4. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements intentionally avoid pinning PyTorch so you can match your environment. Hugging Face models are downloaded
   automatically the first time you run the node.

## Using the nodes

* **SemanticSAMLoader** – selects the SAM checkpoint and the Hugging Face segmentation backbone (OneFormer ADE20K by default).
* **PointPrompt** – helper node to provide click coordinates.
* **SemanticSAMSegment** – runs SAM using the provided prompt, assigns semantic labels, and returns:
  * RGBA previews (`IMAGE` output).
  * Mask tensors (`MASK` output).
  * A JSON string containing metadata for each mask (`STRING` output). Example entry:
    ```json
    {
      "mask_index": 0,
      "label_id": 3,
      "label": "sky",
      "confidence": 0.92,
      "area": 15234,
      "bbox": [10, 24, 512, 256]
    }
    ```

The bundled workflow (`workflow/workflow.json`) demonstrates the loader/segment pipeline.

## Updating existing workflows

Existing workflows continue to function – the node signatures are unchanged apart from the additional metadata output on
`SemanticSAMSegment`. If you do not connect the third output nothing needs to change, but the JSON metadata is ready for storing
in databases or passing into other automation (n8n, Postgres, etc.).

## Troubleshooting

* The first run will download the OneFormer weights from Hugging Face. Ensure the machine has internet access or pre-download the
  models using the `huggingface_hub` CLI.
* GPU execution is recommended for the Semantic-SAM checkpoints. If CUDA is unavailable the segmenter still falls back to CPU,
  but Semantic-SAM itself currently expects a CUDA device.
