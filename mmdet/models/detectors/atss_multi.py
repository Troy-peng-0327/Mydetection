# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
from ..utils import Fusion_strategy


@MODELS.register_module()
class ATSSMulti(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.backbone_t = MODELS.build(backbone)
        self.fuse = Fusion_strategy(neck['out_channels'])

    def extract_feat(self, img):
        img_v, img_t = img
        feat_v = self.backbone(img_v)
        feat_t = self.backbone_t(img_t)
        if self.with_neck:
            feat_v = self.neck(feat_v)
            feat_t = self.neck(feat_t)

        features = []
        for i in range(len(feat_v)):
            fused_feat = self.fuse(feat_v[i], feat_t[i], 'cat')
            features.append(fused_feat)
        return features
