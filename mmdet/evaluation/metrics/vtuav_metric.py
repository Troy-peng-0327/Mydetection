from typing import List, Optional, Sequence, Union

from mmdet.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class VTUAVMetric(CocoMetric):
    """VTUAV evaluation metric."""
    default_prefix: Optional[str] = 'vtuav'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = True,
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
                 use_mp_eval: bool = False) -> None:
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
            collect_device=collect_device, prefix=prefix,
            sort_categories=sort_categories,
            use_mp_eval=use_mp_eval)
