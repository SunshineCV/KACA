_base_ = '../mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(type='LCTBlock', groups=64),
                stages=(False, True, True, True),
                position='after_conv3')
        ]))
