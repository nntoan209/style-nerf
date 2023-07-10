set -e

feature_configs_file="configs/nerf/$1/nerf_synthetic_feature.txt"
style_configs_file="configs/nerf/$1/nerf_synthetic_style.txt"
CUDA_VISIBLE_DEVICES=$2

# train feature
python -W ignore train_feature.py --config $feature_configs_file

# train style
python -W ignore train_style.py --config $style_configs_file
