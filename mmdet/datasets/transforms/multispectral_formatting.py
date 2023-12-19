import numpy as np
from mmengine.structures import InstanceData, PixelData

from mmcv.transforms import to_tensor
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
from .formatting import PackDetInputs


@TRANSFORMS.register_module()
class MultiPackDetInputs(PackDetInputs):
    """Pack the multi inputs data for the detection"""

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img1' in results:
            img1 = results['img1']
            img2 = results['img2']
            if len(img1.shape) < 3:
                img1 = np.expand_dims(img1, -1)
            if len(img2.shape) < 3:
                img2 = np.expand_dims(img2, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img1.flags.c_contiguous:
                img1 = np.ascontiguousarray(img1.transpose(2, 0, 1))
                img1 = to_tensor(img1)
                img2 = np.ascontiguousarray(img2.transpose(2, 0, 1))
                img2 = to_tensor(img2)
            else:
                img1 = to_tensor(img1).permute(2, 0, 1).contiguous()
                img2 = to_tensor(img2).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = [img1, img2]

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
