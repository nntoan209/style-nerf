import torch
import numpy as np
import os, sys
from tqdm import tqdm
import imageio
from torchvision import transforms
from PIL import Image


from StyleRF.opt import config_parser
from StyleRF.renderer import OctreeRender_trilinear_fast
from StyleRF.dataLoader import BlenderDataset
from StyleRF.dataLoader.ray_utils import denormalize_vgg
from StyleRF.models.tensoRF import TensorVMSplit

def get_rays(directions, c2w):
    """
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

@torch.no_grad()
def evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, chunk_size=2048, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=False, style_img=None, device='cuda', infer_type="image"):
    rgb_maps= []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + '/feature', exist_ok=True)
    W, H = test_dataset.img_wh

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        c2w = c2w.float()
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d], 1).reshape(H, W, 6).permute(2,0,1)  # (6,H,W)
        rays = rays.permute(1,2,0).reshape(-1,6) # (H * W, 6)

        feature_map, _, _ = renderer(rays, tensorf, chunk=chunk_size, N_samples=N_samples, ndc_ray=ndc_ray, 
                                white_bg = white_bg, render_feature=True, style_img=style_img, device=device)
        
        feature_map = feature_map.reshape(H, W, 256)[None,...].permute(0,3,1,2)

        recon_rgb = denormalize_vgg(tensorf.decoder(feature_map))
        recon_rgb = recon_rgb.permute(0,2,3,1).clamp(0,1)
        recon_rgb = recon_rgb.reshape(H, W, 3).cpu()
        recon_rgb = (recon_rgb.numpy() * 255).astype('uint8')
        rgb_maps.append(recon_rgb)
        
        if (savePath is not None) and (infer_type=="image"):
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', recon_rgb)

    if infer_type == "video":
        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)

    return 

    
@torch.no_grad()
def generate_video_stylerf(scene, style_image):
    renderer = OctreeRender_trilinear_fast
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    
    args.expname = scene
    args.config = f"../StyleRF/configs/nerf/{scene}/nerf_synthetic_style.txt"
    args.datadir = f"../data/nerf_synthetic/{scene}"
    args.ckpt = f"../StyleRF/log_style/{scene}/{scene}.th"
    args.render_only = 1
    args.render_train = 0
    args.render_test = 0
    args.render_path = 1
    args.chunk_size = 512
    args.rm_weight_mask_thre = 0.0001
    args.rm_weight_mask_thre = 0.0001
    args.n_lamb_sigma = [16,16,16]
    args.n_lamb_sh = [48,48,48]
    args.N_voxel_init = 2097156 # 128**3
    args.N_voxel_final = 27000000 # 300**3
    
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return


    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.change_to_style_mod(device)
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    logfolder = os.path.dirname(args.ckpt)

    trans = transforms.Compose([transforms.Resize(size=(256,256)), transforms.ToTensor()])
    style_img = trans(style_image).cuda()[None, ...]

    if args.render_path:
        test_dataset = BlenderDataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, infer_cfg=None)
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all/', exist_ok=True)
        evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_path_all/',
                N_vis=-1, N_samples=-1, white_bg = test_dataset.white_bg, ndc_ray=ndc_ray, style_img=style_img, device=device, infer_type="video")
        
    return f"../StyleRF/log_style/{scene}/{scene}/imgs_path_all/video.mp4"

@torch.no_grad()
def generate_image_stylerf(scene, style_image, phi, theta):
    
    renderer = OctreeRender_trilinear_fast
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    
    args.expname = scene
    args.datadir = f"../data/nerf_synthetic/{scene}"
    args.ckpt = f"../StyleRF/log_style/{scene}/{scene}.th"
    args.render_only = 1
    args.render_train = 0
    args.render_test = 0
    args.render_path = 1
    args.chunk_size = 512
    args.rm_weight_mask_thre = 0.0001
    args.n_lamb_sigma = [16,16,16]
    args.n_lamb_sh = [48,48,48]
    args.N_voxel_init = 2097156 # 128**3
    args.N_voxel_final = 27000000 # 300**3
    
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)    
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.change_to_style_mod(device)
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    logfolder = os.path.dirname(args.ckpt)

    trans = transforms.Compose([transforms.Resize(size=(256,256)), transforms.ToTensor()])
    style_img = trans(style_image).cuda()[None, ...]

    if args.render_path:
        infer_cfg = {
            "theta": theta,
            "phi": phi
        }
        test_dataset = BlenderDataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, infer_cfg=infer_cfg)
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all/', exist_ok=True)
        evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_path_all/',
                N_vis=-1, N_samples=-1, white_bg = test_dataset.white_bg, ndc_ray=ndc_ray, style_img=style_img, device=device, infer_type="image")
        
    return Image.open(f"../StyleRF/log_style/{scene}/{scene}/imgs_path_all/000.png")
