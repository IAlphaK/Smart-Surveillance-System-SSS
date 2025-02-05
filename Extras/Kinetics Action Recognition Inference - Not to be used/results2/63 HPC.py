_base_ = '../../_base_/default_runtime.py'

# Model configuration with detailed parameters for action detection
model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'
    ),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        resample_rate=8,  # Increased to better capture subtle smoking gestures
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            frozen_stages=2,  # Reduced frozen stages for better fine-tuning
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            frozen_stages=2,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            temporal_pool_mode='max'),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=2304,
            num_classes=3,  # Background + Smoking + Fighting
            multilabel=True,
	topk=(2,),
            dropout_ratio=0.6)),  # Increased dropout for better regularization
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.7,  # Reduced to consider more positive samples
                neg_iou_thr=0.3,
                min_pos_iou=0.7),
            sampler=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

# Dataset configuration
dataset_type = 'AVADataset'
data_root = '/home/i200623/mmaction2/data/custom_ava/rawframes'
ann_file_train = '/home/i200623/mmaction2/data/custom_ava/annotations/new_custom_train.csv'
ann_file_val = '/home/i200623/mmaction2/data/custom_ava/annotations/new_custom_val.csv'
label_file = '/home/i200623/mmaction2/data/custom_ava/annotations/action_list.pbtxt'
proposal_file_train = '/home/i200623/mmaction2/data/custom_ava/annotations/train_gpt_self.pkl'
proposal_file_val = '/home/i200623/mmaction2/data/custom_ava/annotations/val_gpt_self.pkl'

file_client_args = dict(io_backend='disk')

# Enhanced training pipeline with augmentations
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter',
         brightness=0.2,
         contrast=0.2,
         saturation=0.2,
         hue=0.1),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),  # Added fixed resize
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# Validation pipeline without augmentations
val_pipeline = [
    dict(
        type='SampleAVAFrames',
        clip_len=32,
        frame_interval=2,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

# Data loaders configuration
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        custom_classes=[1, 2],  # Smoking and Fighting classes
        num_classes=3))  # Including background class

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
        custom_classes=[1, 2],
        num_classes=3))

test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=None,
    custom_classes=[1, 2],
    num_classes=3)

test_evaluator = val_evaluator

# Optimizer configuration with layer-wise learning rates
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # Base learning rate
        weight_decay=0.05,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=40, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.slow_path': dict(lr_mult=1.0),
        }))

# Two-stage learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=40,
        convert_to_iter_based=True)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_begin=1,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Loss configuration with class weights
loss_cls = dict(
    type='CrossEntropyLoss',
    weight=[1.0, 3.0, 1.0],  # Higher weight for smoking class (index 1)
)

# Default settings
auto_scale_lr = dict(enable=False, base_batch_size=128)
default_hooks = dict(
    logger=dict(interval=20, ignore_last=False),
    checkpoint=dict(interval=1, save_best='auto', max_keep_ckpts=3))

# Logging configuration
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False