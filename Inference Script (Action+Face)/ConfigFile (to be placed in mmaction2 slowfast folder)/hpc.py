_base_ = '../../_base_/default_runtime.py'

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=2304,
            num_classes=3,
            topk=(1,),
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

data_root = '/home/i200623/mmaction2/data/custom_ava/rawframes'
ann_file_train = '/home/i200623/mmaction2/data/custom_ava/annotations/new_custom_train.csv'
ann_file_val = '/home/i200623/mmaction2/data/custom_ava/annotations/new_custom_val.csv'
label_file = '/home/i200623/mmaction2/data/custom_ava/annotations/action_list.pbtxt'
proposal_file_train = '/home/i200623/mmaction2/data/custom_ava/annotations/train_gpt_self.pkl'
proposal_file_val = '/home/i200623/mmaction2/data/custom_ava/annotations/val_gpt_self.pkl'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,  # Increased batch size
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AVADataset',
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        custom_classes=[1, 2],
        num_classes=3))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AVADataset',
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
        custom_classes=[1, 2],
        num_classes=3))
test_dataloader = val_dataloader

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        begin=5,
        end=20,
        by_epoch=True,
        eta_min=1e-6)
]
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# Adjust class weights (Increase weight for Smoking class)
loss_cls = dict(
    type='CrossEntropyLoss', 
    weight=[1.0, 3.0, 1.0]  # Higher weight for Smoking class (index 1)
)

auto_scale_lr = dict(enable=False, base_batch_size=128)
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=None,
    custom_classes=[1, 2],
    num_classes=3)
test_evaluator = val_evaluator

test_cfg = dict(type='TestLoop')

# Distributed Training Settings - Disabled for single GPU
dist_params = None  # Disable distributed training
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

# Single GPU training
runner = dict(type='EpochBasedRunner', max_epochs=40)
