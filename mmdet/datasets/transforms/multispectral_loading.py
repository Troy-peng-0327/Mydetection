# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import os.path as osp
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

import mmengine.fileio as fileio


@TRANSFORMS.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
    """Load an image pair from ``results['img']``."""
    def __init__(self,
                 spectrals=['visible', 'thermal'],
                 day_or_night=None,
                 **kwargs):
        super(LoadImagePairFromFile, self).__init__(**kwargs)
        self.spectrals = spectrals
        self.day_or_night = day_or_night
    
    def transform(self, results):
        data_root, img_name = osp.split(results['img_path'])
        filename1 = osp.join(data_root, self.spectrals[0], img_name)
        filename2 = osp.join(data_root, self.spectrals[1], img_name)
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename1)
                img1_bytes = file_client.get(filename1)
                img2_bytes = file_client.get(filename2)
            else:
                img1_bytes = fileio.get(
                    filename1, backend_args=self.backend_args)
                img2_bytes = fileio.get(
                    filename2, backend_args=self.backend_args)
            img1 = mmcv.imfrombytes(
                img1_bytes, flag=self.color_type, backend=self.imdecode_backend)
            img2 = mmcv.imfrombytes(
                img2_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img1 is not None, f'failed to load image: {filename1}'
        assert img2 is not None, f'failed to load image: {filename2}'
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape[:2]
        results['ori_shape'] = img1.shape[:2]
        return results
