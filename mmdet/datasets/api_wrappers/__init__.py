# Copyright (c) OpenMMLab. All rights reserved.
from .coco_api import COCO, COCOeval, COCOPanoptic
from .cocoeval_mp import COCOevalMP
from .cocoeval_tiny import COCOevalTiny, Params

__all__ = ['COCO', 'COCOeval', 'COCOPanoptic', 'COCOevalMP', 'COCOevalTiny', 'Params']
