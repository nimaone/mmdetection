# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_obb import SingleStageDetector_obb


@DETECTORS.register_module()
class ATSS_obb(SingleStageDetector_obb):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSS_obb, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
