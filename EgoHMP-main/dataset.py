import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from scipy.spatial.transform import Rotation

class EgoDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.fps = config.fps

        self.offline_dir = os.path.join(self.dataroot, 'offline_data_full')
        self.p2_logits_dir = config.p2_logits_dir

        self.affordance_dir = os.path.join(self.dataroot, 'offline_affordance_gt')

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))
        self.df = self.dataset_info[self.dataset_info['training'] == (1 if train else 0)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        scene = row['scene']
        seq = row['sequence_path']
        start_f = int(row['start_frame'])
        t_val = row.get('transformation')
        transform_json = t_val if pd.notna(t_val) else 'transform_info.json'

        seq_name_full = f"{scene}_{seq.replace('/', '_')}_{start_f}"
        seq_dir = os.path.join(self.dataroot, scene, seq)
        sdf_data = torch.load(os.path.join(self.config.dataroot, 'offline_sdf_only', f"{seq_name_full}.pt"),
                              map_location='cpu')
        base_data = torch.load(os.path.join(self.offline_dir, f"{seq_name_full}.pt"), map_location='cpu')
        fv = base_data['fv']
        fm = base_data['fm']
        pts_local_full = base_data['pc_input'].clone()
        label = base_data['label']


        with open(os.path.join(seq_dir, transform_json), 'r') as f:
            t_info = json.load(f)

        scale = t_info['scale']
        m_p2s = np.array(t_info['transformation'])
        m_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
        t_p2s_s = m_p2s.copy()
        t_p2s_s[:3, 3] /= scale
        t_norm_s = m_norm.copy()
        t_norm_s[:3, 3] /= scale
        m_h2w = t_norm_s @ t_p2s_s

        def to_world_trans(pkl):
            t_raw = pkl['trans'].detach().cpu().numpy().reshape(3, 1)
            return (m_h2w[:3, :3] @ t_raw + m_h2w[:3, 3:]).flatten()

        def to_world_rot(pkl):
            r_raw = Rotation.from_rotvec(pkl['orient'].detach().cpu().numpy().flatten()).as_matrix()
            return m_h2w[:3, :3] @ r_raw

        raw_poses = []
        for step in range(16):
            pose_idx = start_f + int(step * 30 / self.fps)
            pkl_path = os.path.join(seq_dir, 'smplx_local', f'{pose_idx}.pkl')

            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    raw_poses.append(pickle.load(f))
            else:
                raw_poses.append(None)

        for i in range(16):
            if raw_poses[i] is None and i > 0 and raw_poses[i - 1] is not None:
                raw_poses[i] = raw_poses[i - 1]
        for i in range(14, -1, -1):
            if raw_poses[i] is None and raw_poses[i + 1] is not None:
                raw_poses[i] = raw_poses[i + 1]

        if raw_poses[5] is None or raw_poses[0] is None:
            return {
                'fv': fv, 'fm': fm, 'target_p': torch.zeros(3), 'label': label,
                'hist_motion': torch.zeros(6, 38), 'hist_traj': torch.zeros(6, 6),
                'gt_motion': torch.zeros(10, 38), 'gt_traj': torch.zeros(10, 6),
                'seq_name': seq_name_full, 'pts_local_full': pts_local_full,
                'udf_grid': sdf_data['sdf_grid'], 'udf_bounds': sdf_data['sdf_bounds'],
                'affordance_gt': torch.zeros(10, 10475)
            }

        p_start = raw_poses[0]
        t_start_w = to_world_trans(p_start)
        r_start_w = to_world_rot(p_start)
        p_center = raw_poses[5]
        t_center_w = to_world_trans(p_center)
        r_center_w = to_world_rot(p_center)
        pts_xyz = pts_local_full[:, :3].numpy()  # [N, 3]
        pts_world = (r_start_w @ pts_xyz.T).T + t_start_w
        pts_aligned = (r_center_w.T @ (pts_world - t_center_w).T).T
        pts_local_full[:, :3] = torch.from_numpy(pts_aligned).float()

        if self.train:
            gt_best_idx = torch.argmax(label)
            target_p = pts_local_full[gt_best_idx, :3]
        else:
            p2_file = f"{index:04d}_{seq_name_full}.pth"
            p2_path = os.path.join(self.p2_logits_dir, p2_file)

            if os.path.exists(p2_path):
                p2_res = torch.load(p2_path, map_location='cpu')
                best_idx = torch.argmax(p2_res['logits'])
                target_p = pts_local_full[best_idx, :3]
            else:
                gt_best_idx = torch.argmax(label)
                target_p = pts_local_full[gt_best_idx, :3]

        t0_w = t_center_w
        r0_w = r_center_w
        hist_motion_list = []
        future_motion_list = []

        for step in range(16):
            pk = raw_poses[step]
            tk_w = to_world_trans(pk)
            rk_w = to_world_rot(pk)
            tk_local = (r0_w.T @ (tk_w - t0_w).reshape(3, 1)).flatten()
            rk_local_mat = r0_w.T @ rk_w
            rk_local_vec = Rotation.from_matrix(rk_local_mat).as_rotvec()
            latent = pk['latent'].cpu().detach().flatten()
            motion_k = torch.cat([
                torch.from_numpy(rk_local_vec).float(),
                torch.from_numpy(tk_local).float(),
                latent
            ])
            if step < 6:
                hist_motion_list.append(motion_k)
            else:
                future_motion_list.append(motion_k)
        hist_motion = torch.stack(hist_motion_list)  # [6, 38]
        hist_traj = hist_motion[:, :6]  # [6, 6]
        future_motion = torch.stack(future_motion_list)  # [10, 38]
        gt_traj = future_motion[:, :6]  # [10, 6]
        affordance_path = os.path.join(self.affordance_dir, f"{seq_name_full}_affordance.pt")
        if os.path.exists(affordance_path):
            affordance_gt_full = torch.load(affordance_path, map_location='cpu')  # [16, 10475]
        else:
            affordance_gt_full = torch.zeros(16, 10475)
        affordance_gt = affordance_gt_full[6:, :]  # [10, 10475]
        return {
            'fv': fv,
            'fm': fm,
            'target_p': target_p,
            'label': label,
            'hist_motion': hist_motion,
            'hist_traj': hist_traj,
            'gt_motion': future_motion,
            'gt_traj': gt_traj,
            'seq_name': seq_name_full,
            'pts_local_full': pts_local_full,
            'udf_grid': sdf_data['sdf_grid'],
            'udf_bounds': sdf_data['sdf_bounds'],
            'affordance_gt': affordance_gt
        }