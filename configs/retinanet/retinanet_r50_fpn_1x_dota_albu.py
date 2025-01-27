import numpy as np
model = dict(
    type='RetinaNet_obb',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead_obb',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=2000))    



dataset_type = 'dotaDataset'
data_root = '/content/DOTA_ssplit_600_150/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(                     
     type='Resize',
        height=800, width=800, p=1),              
    dict(
        type='RandomSizedCrop',
         min_max_height=(600, 800), height=800, width=800, p=0.5),
    # dict(
    #     type='Rotate',       
    #         limit=list(np.arange(-90, 90+16, 15)),
    #       interpolation=1, border_mode=0, p=0.8),    
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.2,
        rotate_limit=list(np.arange(-90, 90+16, 15)),
        interpolation=1,
        border_mode=0,
        p=.9),
    
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.9),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=.2,
                sat_shift_limit=.2,
                val_shift_limit=.2,
                p=0.9)
        ],
        p=0.2),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#     dict(type='ChannelShuffle', p=0.1),
    dict(type='ToGray', p=0.05),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=10,
            min_visibility=0.6,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize',
         mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='dotaDataset',
        ann_file='train/DOTA_train.json',
        img_prefix='train/images',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type='dotaDataset',
        ann_file='val/DOTA_val.json',
        img_prefix='val/images',
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type='dotaDataset',
        ann_file='train/DOTA_train.json',
        img_prefix='train/images',
        pipeline=test_pipeline,
        data_root=data_root))


# evaluation = None
evaluation = dict(interval=1, metric='mAP')


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])


runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/content/mmdetection/checkpoint/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '/content/drive/MyDrive/tutorial_exps1'
seed = 0
gpu_ids = range(0, 1)
