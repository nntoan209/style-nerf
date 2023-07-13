_base_ = '../default.py'

expname = 'lego'
basedir = '../UPST_NeRF/logs/nerf_synthetic'

data = dict(
    datadir='../data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

