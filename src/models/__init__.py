"""
Models package for SeaClear underwater marine debris instance segmentation.
Exposes:
- M1: MaskRTDETRBaseline
- M2: MaskRTDETR_IAM
- M3: MaskRTDETR_YOLACTProto
- M4: ProposedMaskRTDETR_IAM_YOLACT
- DETRCriterion
- HungarianMatcher
"""

from .backbone import ResNetBackbone
from .hybrid_encoder import HybridEncoder
from .detr_decoder import TransformerDecoderLayer, MLP, HungarianMatcher, DETRCriterion, box_cxcywh_to_xyxy, generalized_box_iou
from .iam import InstanceActivationMapModule
from .yolact_proto import Protonet, YOLACTPrototypeBranch
from .m1_mask_rtdetr import MaskRTDETRBaseline
from .m2_iam_rtdetr import MaskRTDETR_IAM
from .m3_proto_rtdetr import MaskRTDETR_YOLACTProto
from .m4_proposed import ProposedMaskRTDETR_IAM_YOLACT

__all__ = [
    "ResNetBackbone",
    "HybridEncoder",
    "TransformerDecoderLayer",
    "MLP",
    "HungarianMatcher",
    "DETRCriterion",
    "box_cxcywh_to_xyxy",
    "generalized_box_iou",
    "InstanceActivationMapModule",
    "Protonet",
    "YOLACTPrototypeBranch",
    "MaskRTDETRBaseline",
    "MaskRTDETR_IAM",
    "MaskRTDETR_YOLACTProto",
    "ProposedMaskRTDETR_IAM_YOLACT",
]
