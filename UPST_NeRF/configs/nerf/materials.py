_base_ = '../default.py'

expname = 'materials'
basedir = '../UPST_NeRF/logs/nerf_synthetic'

data = dict(
    datadir='../data/nerf_synthetic/materials',
    dataset_type='blender',
    white_bkgd=True,
)

