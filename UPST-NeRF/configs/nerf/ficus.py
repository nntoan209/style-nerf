_base_ = '../default.py'

expname = 'ficus'
basedir = '../UPST_NeRF/logs/nerf_synthetic'

data = dict(
    datadir='../data/nerf_synthetic/ficus',
    dataset_type='blender',
    white_bkgd=True,
)

