auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/'
dataset_type = 'RGBTDronePersonDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=80,
        stacked_convs=4,
        type='ATSSHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            115.37,
            121.82,
            122.63,
        ],
        mean2=[
            93.1,
            93.1,
            93.1,
        ],
        pad_size_divisor=32,
        std=[
            85.13,
            89.01,
            88.27,
        ],
        std2=[
            50.24,
            50.24,
            50.24,
        ],
        type='MutliDetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='ATSSMulti')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val_thermal.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root=
        '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/',
        pipeline=[
            dict(backend_args=None, type='LoadImagePairFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='MultiResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='MultiPackDetInputs'),
        ],
        test_mode=True,
        type='RGBTDronePersonDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/val_thermal.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='RGBTDronePersonMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImagePairFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='MultiResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='MultiPackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='train_thermal.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root=
        '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImagePairFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='MultiResize'),
            dict(prob=0.5, type='MultiRandomFlip'),
            dict(type='MultiPackDetInputs'),
        ],
        type='RGBTDronePersonDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImagePairFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='MultiResize'),
    dict(prob=0.5, type='MultiRandomFlip'),
    dict(type='MultiPackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val_thermal.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root=
        '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/',
        pipeline=[
            dict(backend_args=None, type='LoadImagePairFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='MultiResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='MultiPackDetInputs'),
        ],
        test_mode=True,
        type='RGBTDronePersonDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/Users/pengpeiran/Desktop/PhD_documents/05代码/Datasets/RGBTDronePerson/val_thermal.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='RGBTDronePersonMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dir/atss_r50_fpn/rgbtdroneperson'