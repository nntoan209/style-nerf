_base_ = '../default.py'

expname = 'chair'
basedir = '../UPST_NeRF/logs/nerf_synthetic'

data = dict(
    datadir='../data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
)

