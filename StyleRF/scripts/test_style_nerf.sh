set -e

expname=$1
style_image_path=$2
CUDA_VISIBLE_DEVICES=$3

config_file="configs/nerf/$1/nerf_synthetic_style.txt"
data_dir="../data/nerf_synthetic/$1"

python train_style.py --config $config_file \
--datadir $data_dir \
--expname $expname \
--ckpt log_style/$expname/$expname.th \
--style_img $style_image_path \
--render_only 1 \
--render_train 0 \
--render_test 1 \
--render_path 0 \
--chunk_size 512 \
--rm_weight_mask_thre 0.0001
