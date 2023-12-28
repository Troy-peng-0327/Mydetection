from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .coco import CocoDataset


@DATASETS.register_module()
class VTUAVDetDataset(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True