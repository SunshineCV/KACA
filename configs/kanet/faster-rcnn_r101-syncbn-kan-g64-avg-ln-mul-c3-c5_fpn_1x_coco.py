_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
        backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101'),
        plugins=[
            dict(
                cfg=dict(type='KACABlock', use_layernorm=True, groups=64, num_grids=12, grid_max=2, grid_min=-2, fusion='mul', pooling_type='avg'),
                stages=(False, True, True, True),
                position='after_conv3')
        ]))
