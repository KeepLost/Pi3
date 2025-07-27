import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge,recover_intrinsics,reconstruct_pointcloud
from pi3.models.pi3 import Pi3

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--reproj",action='store_true',help="Calculate intrinsics and reproject points to 3D space.")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs,img_masks = load_images_as_tensor(args.data_path, interval=args.interval) # (N, 3, H, W)
    imgs=imgs.to(device)
    if img_masks is not None:
        img_masks.to(device)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    # 4. process mask
    confs=res['conf']
    if img_masks is not None:
        image_masks=img_masks.permute(0,2,3,1)[None].to(device)
        confs=torch.where(image_masks>=0.5,confs,-100)
    masks = torch.sigmoid(confs[..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    if args.reproj:
        K = recover_intrinsics(res['local_points'], masks.unsqueeze(0).unsqueeze(-1).float(), conf_thresh=0.1)
        recon_pts=reconstruct_pointcloud(
            res['local_points'],res['camera_poses'], K
        )
        mask3d = (masks.unsqueeze(0).float() > 0.1)
        diff = (recon_pts - res['points']).norm(dim=-1)
        error_map = diff * mask3d.float()
        mean_error = error_map.sum() / mask3d.sum().clamp(min=1)
        print(f"Mean reprojection error (in world units): {mean_error:.6f}")
    print("Done.")