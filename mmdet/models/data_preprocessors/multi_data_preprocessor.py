import math
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor, stack_batch
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

try:
    import skimage
except ImportError:
    skimage = None


@MODELS.register_module()
class MutliDetDataPreprocessor(ImgDataPreprocessor):
    """Multi Image pre-processor for detection tasks."""

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 mean2: Sequence[Number] = None,
                 std2: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor
        assert (mean2 is None) == (std2 is None), (
            'mean2 and std2 should be both None or tuple')
        if mean2 is not None:
            assert len(mean2) == 3 or len(mean2) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean2)} values')
            assert len(std2) == 3 or len(std2) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std2)} values')  # type: ignore
            self._enable_normalize = True
            self.register_buffer('mean2',
                                 torch.tensor(mean2).view(-1, 1, 1), False)
            self.register_buffer('std2',
                                 torch.tensor(std2).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)

        data = self.cast_data(data)  # type: ignore
        _batch_inputs1, _batch_inputs2 = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs1, torch.Tensor):
            batch_inputs1 = []
            batch_inputs2 = []
            for _batch_input1, _batch_input2 in zip(_batch_inputs1, _batch_inputs2):
                # channel transform
                if self._channel_conversion:
                    _batch_input1 = _batch_input1[[2, 1, 0], ...]
                    _batch_input2 = _batch_input2[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input1 = _batch_input1.float()
                _batch_input2 = _batch_input2.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input1.dim(
                        ) == 3 and _batch_input1.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input1.shape}')
                    _batch_input1 = (_batch_input1 - self.mean) / self.std
                    if self.mean2.shape[0] == 3:
                        assert _batch_input2.dim(
                        ) == 3 and _batch_input2.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input2.shape}')
                    _batch_input2 = (_batch_input2 - self.mean2) / self.std2
                batch_inputs1.append(_batch_input1)
                batch_inputs2.append(_batch_input2)
            # Pad and stack Tensor.
            batch_inputs1 = stack_batch(batch_inputs1, self.pad_size_divisor,
                                       self.pad_value)
            batch_inputs2 = stack_batch(batch_inputs2, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs1, torch.Tensor):
            assert _batch_inputs1.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs1.shape}')
            if self._channel_conversion:
                _batch_inputs1 = _batch_inputs1[:, [2, 1, 0], ...]
                _batch_inputs2 = _batch_inputs2[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs1 = _batch_inputs1.float()
            _batch_inputs2 = _batch_inputs2.float()
            if self._enable_normalize:
                _batch_inputs1 = (_batch_inputs1 - self.mean) / self.std
                _batch_inputs2 = (_batch_inputs2 - self.mean2) / self.std2
            h, w = _batch_inputs1.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs1 = F.pad(_batch_inputs1, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
            batch_inputs2 = F.pad(_batch_inputs2, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = [batch_inputs1, batch_inputs2]
        data.setdefault('data_samples', None)

        inputs1, inputs2 = data['inputs']
        data_samples = data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs1[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': [inputs1, inputs2], 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs1, _batch_inputs2 = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs1, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs1:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs1, torch.Tensor):
            assert _batch_inputs1.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs1.shape}')
            pad_h = int(
                np.ceil(_batch_inputs1.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs1.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs1.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)