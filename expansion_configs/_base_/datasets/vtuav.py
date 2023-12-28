# dataset settings
dataset_type = 'VTUAVDetDataset'
data_root = '../../Datasets/VTUAV/'

backend_args = None

train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=['rgb', 'ir'], backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', scale=(640, 512), keep_ratio=True),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='MultiPackDetInputs')
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=['rgb', 'ir'], backend_args=backend_args),
    dict(type='MultiResize', scale=(640, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_ir.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_ir.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='VTUAVMetric',
    ann_file=data_root + 'val_ir.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator