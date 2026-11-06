from .sift_block import SiftBlock
from .sift_model import SiftBackbone, SiftConfig, SiftLM, count_params
from .transformer import TransformerBackbone, TransformerConfig, TransformerLM
from .siftx import SiftXBackbone, SiftXLM, build_siftx


def build_lm(model_type, **kwargs):
    if model_type == "sift":
        return SiftLM(SiftConfig(**kwargs))
    if model_type == "transformer":
        return TransformerLM(TransformerConfig(**kwargs))
    if model_type == "siftx":
        return SiftXLM(build_siftx(kwargs.pop("variant", "siftx"), **kwargs))
    raise ValueError(f"unknown model type: {model_type}")


def build_backbone(model_type, **kwargs):
    if model_type == "sift":
        return SiftBackbone(SiftConfig(**kwargs))
    if model_type == "transformer":
        return TransformerBackbone(TransformerConfig(**kwargs))
    if model_type == "siftx":
        return SiftXBackbone(build_siftx(kwargs.pop("variant", "siftx"), **kwargs))
    raise ValueError(f"unknown model type: {model_type}")
