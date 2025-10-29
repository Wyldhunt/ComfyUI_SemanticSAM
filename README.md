# ComfyUI SemanticSAM (modernised)

This project provides ComfyUI nodes that pair [Segment Anything](https://segment-anything.com/) mask generation with modern
Hugging Face semantic segmentation models. The loader/segment workflow mirrors the original SemanticSAM experience while keeping
the implementation entirely in pure PyTorch.

## What's new

* ✅ Pure PyTorch inference path – no compiled extensions or legacy dependencies.
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

2. Download the Segment Anything checkpoints into `custom_nodes/ComfyUI_SemanticSAM/ckpt`:

   | Name | Download |
   | --- | --- |
   | SAM ViT-H | [model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
   | SAM ViT-L | [model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
   | SAM ViT-B | [model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

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

* The first run will download the OneFormer weights from Hugging Face. Ensure the machine has internet access or pre-download the models using the `huggingface_hub` CLI.
* GPU execution is recommended for the SAM checkpoints. CPU execution is available but significantly slower for high-resolution images.
