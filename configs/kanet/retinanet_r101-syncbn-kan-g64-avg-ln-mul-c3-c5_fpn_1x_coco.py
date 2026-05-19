_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        plugins=[
        dict(
            cfg=dict(type='KACABlock', use_layernorm=True, groups=64, num_grids=12, grid_max=2, grid_min=-2, fusion='mul', pooling_type='avg'),
            stages=(False, True, True, True),
            position='after_conv3')
        ],
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
