_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        plugins=[
        dict(
            cfg=dict(type='GCTBlock'),
            stages=(False, True, True, True),
            position='after_conv3')
        ],
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
