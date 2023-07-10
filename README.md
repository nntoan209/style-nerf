# style-nerf

## StyleRF


Download data from * [data](https://drive.google.com/file/d/1AKFrdwtQLJ7wPKbdJgNaW-SXJuogKcDO/view?usp=sharing) , unzip and put in the same folder with StyleRF (data and StyleRF are in the same folder)
Download checkpoints for reconstructing density field from * [log](https://drive.google.com/file/d/1zYQFRoKt2UZ3LTIQ1u98zmJ-3jE0GsD9/view?usp=sharing), unzip and put under StyleRF folder (log is under StyleRF)

```
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard plyfile
cd StyleRF
```

There are 8 scences in nerf_synthetic data (chair, drums, hotdog, lego, materials, ship, ficus, mic), need to train a different model for each scene

To train model for a scene, run:

```
sh scripts/train_nerf.sh [scene] [GPU ID] 
```

where scene is 1 of the 8 scene, GPU ID is the id of GPU to use

After training the model, to perform style transfer with custom style image, run:

```
sh scripts/test_style_nerf.sh [scene] [path to the style image] [GPU ID]
```

The generated images and video with style transfer are put under log_style/[scene]