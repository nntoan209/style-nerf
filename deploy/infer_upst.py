import random
import numpy as np
import torch
import argparse, mmengine, imageio
import os
from tqdm import tqdm, trange
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('../')
from UPST_NeRF.lib.load_data import load_data
from UPST_NeRF.lib import utils, upst, upstmpi

def seed_everything():
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)
    
def upst_config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default="./configs/nerf/chair.py",
    # parser.add_argument('--config', default="./configs/llff/fern.py",
                        help='config file path')
    parser.add_argument("--style_img",default="./style_images/IMG_20210921_192826.jpg",
                        help='style_img file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--style_base_root",default="../data/")
    parser.add_argument("--render_style", action='store_true',default=False)
    parser.add_argument("--render_geometry", action='store_true',default=True)
    parser.add_argument("--render_only", action='store_true',default=False,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',default=False)
    parser.add_argument("--render_train", action='store_true',default=False)
    parser.add_argument("--render_video", action='store_true',default=False)
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true',default=True)
    parser.add_argument("--eval_lpips_alex", action='store_true',default=False)
    parser.add_argument("--eval_lpips_vgg", action='store_true',default=False)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=1e6,
                        help='frequency of weight ckpt saving')
    
    # for custom input view
    parser.add_argument("--render_single", action="store_true", default=False)
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    return parser

def load_everything(cfg, infer_cfg=None):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data, infer_cfg)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

@torch.no_grad()
def render_viewpoints_style(style_img, model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, cfg=None, device=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    model.eval()
    print(style_img)
    if isinstance(style_img, str):
        style_images = Image.open(os.path.join(style_img)).convert('RGB')
    else:
        style_images = style_img
        
    data_transform_view = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ])

    style_embedded = data_transform_view(style_images)
    style_embedded = style_embedded.to(device)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = upst.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, torch.unsqueeze(style_embedded, 0), **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(100*100, 0), rays_d.split(100*100, 0), viewdirs.split(100*100, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }


        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        rgbs.append(rgb)
        depths.append(depth)
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)

    return rgbs, depths
    
def generate_image_upst(scene, style_image, phi, theta):
    parser = upst_config_parser()
    args = parser.parse_args()
    args.config = f"../UPST_NeRF/configs/nerf/{scene}.py"
    args.render_only = True
    args.render_style = True
    args.render_single = True
    
    cfg = mmengine.Config.fromfile(args.config)
    cfg.basedir = "../UPST_NeRF/logs/nerf_synthetic"
    
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    
    infer_cfg = {
        'type': "single",
        "phi": phi,
        "theta": theta
    }
    data_dict = load_everything(cfg=cfg, infer_cfg=infer_cfg)
      
    if args.render_style:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_style_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = upstmpi.UPSTMPI_DirectGO
        else:
            model_class = upst.UPST_DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

        # render single image
        if args.render_single:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_single_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints_style(
                    style_img=style_image,
                    render_poses=data_dict['render_poses'],
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    render_factor=args.render_video_factor,
                    savedir=testsavedir,
                    cfg=cfg,
                    device=device,
                    **render_viewpoints_kwargs)
            
            return Image.open(f"../UPST_NeRF/logs/nerf_synthetic/{scene}/render_single_fine_style_last/000.png")
            
            
def generate_video_upst(scene, style_image):
    parser = upst_config_parser()
    args = parser.parse_args()
    args.config = f"../UPST_NeRF/configs/nerf/{scene}.py"
    args.render_only = True
    args.render_style = True
    args.render_video = True
    
    cfg = mmengine.Config.fromfile(args.config)
    cfg.basedir = "../UPST_NeRF/logs/nerf_synthetic"
    
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    
    infer_cfg = {'type': "video"}
    data_dict = load_everything(cfg=cfg, infer_cfg=infer_cfg)
      
    if args.render_style:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_style_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = upstmpi.UPSTMPI_DirectGO
        else:
            model_class = upst.UPST_DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }
   
            
        # render video
        if args.render_video:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints_style(
                    style_img=style_image,
                    render_poses=data_dict['render_poses'],
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    render_factor=args.render_video_factor,
                    savedir=testsavedir,
                    cfg=cfg,
                    device=device,
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            
            return f"../UPST_NeRF/logs/nerf_synthetic/{scene}/render_video_fine_style_last/video.rgb.mp4"
