import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar

from lib.vis.renderer import Renderer, get_global_cameras, get_global_cameras_static

def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world'][:, :3]),
            betas=tt(results[sid]['betas']),
            transl=tt(results[sid]['trans_world']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        
        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)
        
        if vis_global:
            # render the global coordinate
            if frame_i in results[sid]['frame_ids']:
                frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9
                
                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
            
            try: img = np.concatenate((img, img_glob), axis=1)
            except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()


def _lerp_colors(c0, c1, t):
    """c0, c1: RGB in [0,1]. t: array of interpolation fractions in [0,1]."""
    c0, c1 = np.array(c0), np.array(c1)
    return c0[None, :] + (c1 - c0)[None, :] * t[:, None]


def render_trajectory_snapshot(cfg, results, output_pth, smpl,
                                num_ghosts=10,
                                out_size=(1920, 1080),
                                color_start=(0.15, 0.35, 0.90),   # blue
                                color_end=(0.90, 0.20, 0.15),     # red
                                filename='trajectory.png'):
    """Render a single static top-down/angled overview image of the world-space
    trajectory: a ground plane sized to the whole motion extent, a time-gradient
    path line tracing the root position, and `num_ghosts` full-body mesh instances
    evenly spaced across the sequence.
    """
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    # Visualize the subject that appears longest, matching run_vis_on_demo's convention
    n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
    sid = max(n_frames, key=n_frames.get)

    # World-space SMPL vertices for the entire sequence
    global_output = smpl.get_output(
        body_pose=tt(results[sid]['pose_world'][:, 3:]),
        global_orient=tt(results[sid]['pose_world'][:, :3]),
        betas=tt(results[sid]['betas']),
        transl=tt(results[sid]['trans_world']))
    verts_glob = global_output.vertices.cpu()
    floor_offset = verts_glob[..., 1].min()
    verts_glob[..., 1] -= floor_offset
    cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
    sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * 1.5

    # Floor-align the path to match the meshes
    path_world = results[sid]['trans_world'].copy()
    path_world[:, 1] -= floor_offset.item()

    # Video-independent renderer + ground plane
    W, H = out_size
    focal_length = (W ** 2 + H ** 2) ** 0.5
    renderer = Renderer(W, H, focal_length, cfg.DEVICE, smpl.faces)
    renderer.set_ground(scale, cx.item(), cz.item())
    renderer.reset_bbox()

    # Single static camera framing the whole trajectory
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob, cfg.DEVICE, distance=scale * 1.5)

    # Evenly-spaced ghost frames
    T = verts_glob.shape[0]
    n_ghosts = min(num_ghosts, T)
    ghost_idxs = np.linspace(0, T - 1, n_ghosts).round().astype(int)
    ghost_verts = verts_glob[ghost_idxs].to(cfg.DEVICE)

    # Time-gradient colors
    ghost_colors = _lerp_colors(color_start, color_end, ghost_idxs / max(T - 1, 1))
    path_colors = _lerp_colors(color_start, color_end, np.linspace(0, 1, T))

    # Render ghosts + ground in one pass
    faces = renderer.faces.clone().squeeze(0)
    colors = torch.from_numpy(ghost_colors).float().to(cfg.DEVICE)
    colors = torch.cat([colors, torch.ones(n_ghosts, 1, device=cfg.DEVICE)], dim=1)
    cameras = renderer.create_camera(global_R, global_T)
    image = renderer.render_with_ground(ghost_verts, faces, colors, cameras, global_lights)
    image = np.ascontiguousarray(image)

    # Overlay the gradient path line, projected with the same camera
    pts = torch.from_numpy(path_world).float().to(cfg.DEVICE)
    screen_pts = cameras.transform_points_screen(pts, image_size=renderer.image_sizes)[..., :2]
    screen_pts = screen_pts.cpu().numpy()

    for i in range(len(screen_pts) - 1):
        p0 = (int(screen_pts[i][0]), int(screen_pts[i][1]))
        p1 = (int(screen_pts[i + 1][0]), int(screen_pts[i + 1][1]))
        color = tuple(int(c * 255) for c in path_colors[i])
        cv2.line(image, p0, p1, color, thickness=3, lineType=cv2.LINE_AA)

    out_path = osp.join(output_pth, filename)
    imageio.imwrite(out_path, image)
    return out_path