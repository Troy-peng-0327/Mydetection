# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP, COCOevalTiny, Params
from mmdet.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class RGBTDronePersonMetric(CocoMetric):
    """RGBT Drone Person evaluation metric based on COCO metric."""
    default_prefix: Optional[str] = 'rgbtdroneperson'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False,
                 eval_standard: str = 'tiny',
                 ignore_uncertain: bool = False,
                 use_iod_for_crowd: bool = False) -> None:
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            file_client_args=file_client_args,
            backend_args=backend_args,
            collect_device=collect_device, 
            prefix=prefix,
            sort_categories=sort_categories,
            use_mp_eval=use_mp_eval)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .25, 0.75, int(np.round((0.75 - .25) / .25)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.eval_standard = eval_standard
        self.ignore_uncertain = ignore_uncertain
        self.use_iod_for_crowd = use_iod_for_crowd

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            Params.EVAL_STRANDARD = self.eval_standard
            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                if self.eval_standard == 'tiny':
                    coco_eval = COCOevalTiny(self._coco_api, coco_dt, iou_type, ignore_uncertain=True, use_iod_for_ignore=True)
                else:
                    coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            if self.eval_standard == 'coco':
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
            if self.eval_standard == 'tiny':
                coco_metric_names = {
                    'mAP_25': 0,
                    'mAP_25_tiny': 1,
                    'mAP_25_tiny1': 2,
                    'mAP_25_tiny2': 3,
                    'mAP_25_tiny3': 4,
                    'mAP_25_small': 5,
                    'mAP_25_reasonable': 6,
                    'mAP_50': 7,
                    'mAP_50_tiny': 8,
                    'mAP_50_tiny1': 9,
                    'mAP_50_tiny2': 10,
                    'mAP_50_tiny3': 11,
                    'mAP_50_small': 12,
                    'mAP_50_reasonable': 13,
                    'mAP_75': 14,
                    'mAP_75_tiny': 15,
                    'mAP_75_tiny1': 16,
                    'mAP_75_tiny2': 17,
                    'mAP_75_tiny3': 18,
                    'mAP_75_small': 19,
                    'mAP_75_reasonable': 20,                    
                }

            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        # AP25
                        precision = precisions[0, :, idx, 0, -1]
                        # AP
                        # precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    # headers = [
                    #     'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                    #     'mAP_m', 'mAP_l'
                    # ]
                    headers = ['category', 'AP25'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    # metric_items = [
                    #     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    # ]
                    metric_items = [
                        'mAP_75', 'mAP_25', 'mAP_50', 'mAP_50_tiny', 'mAP_50_tiny1', 'mAP_50_tiny2', 'mAP_50_tiny3', 'mAP_50_small',
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                # ap = coco_eval.stats[:6]
                ap = coco_eval.stats[:8]    
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f} '
                            f'{ap[7]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
