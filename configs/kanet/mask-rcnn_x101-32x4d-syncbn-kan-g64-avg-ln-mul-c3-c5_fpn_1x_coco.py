_base_ = '../mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        plugins=[
            dict(
                cfg=dict(type='FastKanBlock', use_layernorm=True, groups=64, grid_min=-2, grid_max=2, fusion='mul', pooling_type='avg'),
                stages=(False, True, True, True),
                position='after_conv3')
        ]))
