from .node import *
import os.path

custom_nodes_path = os.path.dirname(os.path.abspath(__file__))

NODE_CLASS_MAPPINGS = {
    'SemanticSAMLoader': SemanticSAMLoader,
    'SemanticSAMSegment': SemanticSAMSegment,
    'PointPrompt': PointPrompt,
    'StringToBBox': StringToBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticSAMLoader": "Semantic SAM Model Loader",
    "SemanticSAMSegment": "Semantic SAM Segment",
    "PointPrompt": "Point Prompt",
    "StringToBBox": "String to BBOX",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
