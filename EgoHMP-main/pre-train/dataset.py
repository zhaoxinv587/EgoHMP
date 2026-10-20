import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class EgoMotionPretrainDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.input_seq_len = config.input_seq_len if hasattr(config, 'input_seq_len') else 6
        self.fps = config.fps if hasattr(config, 'fps') else 2

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sequences_path_list = []
        self.scenes_path_list = []
        self.poses_path_list = []
        self.video_start_list = []

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))

        self.parse_massive_data_info()

        folder = f"Cache_{self.input_seq_len}F_FPS{self.fps}"
        split = "train" if train else "val"
        self.cache_dir = os.path.join(self.dataroot, folder, split)

        if not os.path.exists(os.path.join(self.cache_dir, "fv_full.pth")) or \
                not os.path.exists(os.path.join(self.cache_dir, "motion_full.pth")):
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocess_and_cache()

        self.fv_full = torch.load(os.path.join(self.cache_dir, "fv_full.pth"))  # [N, 6, 1024]
        self.motion_full = torch.load(os.path.join(self.cache_dir, "motion_full.pth"))  # [N, 6, 12]


    def __len__(self):
        return len(self.fv_full)

    def parse_massive_data_info(self):

        frame_interval = int(30 / self.fps)
        window_span = (self.input_seq_len - 1) * frame_interval  # 75 帧
        blacklist_frames_per_seq = {}
        for _, row in self.dataset_info.iterrows():
            if row['training'] == 0:
                seq = row['sequence_path']
                if seq not in blacklist_frames_per_seq:
                    blacklist_frames_per_seq[seq] = set()
                blacklist_frames_per_seq[seq].update(range(int(row['start_frame']) - 15, int(row['end_frame']) + 16))

        unique_videos = self.dataset_info[['scene', 'sequence_path']].drop_duplicates()

        stride = 10
        total_count = 0

        for _, row in unique_videos.iterrows():
            scene = row['scene']
            seq = row['sequence_path']
            pv_dir = os.path.join(self.dataroot, scene, seq, 'PV')

            if not os.path.exists(pv_dir): continue

            pv_files = [f for f in os.listdir(pv_dir) if f.endswith('.png')]
            total_frames = len(pv_files)

            for slide_idx in range(0, total_frames - window_span, stride):

                sample_frames = [slide_idx + j * frame_interval for j in range(self.input_seq_len)]

                is_blocked = False
                if seq in blacklist_frames_per_seq:
                    for f_idx in sample_frames:
                        if f_idx in blacklist_frames_per_seq[seq]:
                            is_blocked = True
                            break

                if is_blocked:
                    continue

                self.poses_path_list.append(slide_idx)
                self.sequences_path_list.append(seq)
                self.scenes_path_list.append(scene)
                self.video_start_list.append(0)
                total_count += 1


    def absolute_to_relative_motion(self, root_trans, root_orient_mats):
        seq_len = root_trans.shape[0]
        rel_motion = np.zeros((seq_len, 12), dtype=np.float32)
        t_0 = root_trans[0]
        R_0_inv = np.linalg.inv(root_orient_mats[0])

        for i in range(seq_len):
            delta_t = root_trans[i] - t_0
            rel_t = R_0_inv @ delta_t
            rel_R = R_0_inv @ root_orient_mats[i]
            rel_motion[i, :3] = rel_t
            rel_motion[i, 3:] = rel_R.flatten()
        return rel_motion

    def preprocess_and_cache(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
        dinov2.eval()

        all_fv = []
        all_motion = []
        error_printed = False
        dir_cache = {}

        for i in tqdm(range(len(self.poses_path_list)), desc="Data Compression"):
            scene = self.scenes_path_list[i]
            seq = self.sequences_path_list[i]
            ego_idx = self.poses_path_list[i]
            video_start = self.video_start_list[i]

            seq_dir = os.path.join(self.dataroot, scene, seq)
            pv_dir = os.path.join(seq_dir, 'PV')
            smpl_dir = os.path.join(seq_dir, 'smplx_local')

            if seq_dir not in dir_cache:
                try:
                    pv_files = sorted([f for f in os.listdir(pv_dir) if f.endswith('.png')])
                    smpl_files = sorted([f for f in os.listdir(smpl_dir) if f.endswith('.pkl')])
                    dir_cache[seq_dir] = {'pv': pv_files, 'smpl': smpl_files}
                except Exception as e:
                    if not error_printed:
                        print(f"error: {seq_dir} ({e})")
                        error_printed = True
                    continue

            pv_files = dir_cache[seq_dir]['pv']
            smpl_files = dir_cache[seq_dir]['smpl']

            img_batch = []
            root_trans_list = []
            root_orient_list = []
            valid_slice = True

            for f in range(self.input_seq_len):
                pose_idx = ego_idx + int(f * 30 / self.fps)
                local_idx = pose_idx - video_start
                actual_img_path = None
                if pose_idx < len(pv_files):
                    actual_img_path = os.path.join(pv_dir, pv_files[pose_idx])
                elif 0 <= local_idx < len(pv_files):
                    actual_img_path = os.path.join(pv_dir, pv_files[local_idx])

                if actual_img_path and os.path.exists(actual_img_path):
                    img = Image.open(actual_img_path).convert('RGB')
                    img_batch.append(self.transform(img))
                else:
                    valid_slice = False
                    break

                actual_pkl_path = None
                pkl_candidates = [
                    os.path.join(smpl_dir, f'{pose_idx}.pkl'),
                    os.path.join(smpl_dir, f'{pose_idx:06d}.pkl'),
                    os.path.join(smpl_dir, f'{local_idx}.pkl')
                ]

                for path in pkl_candidates:
                    if os.path.exists(path):
                        actual_pkl_path = path
                        break

                if actual_pkl_path is None and pose_idx < len(smpl_files):
                    actual_pkl_path = os.path.join(smpl_dir, smpl_files[pose_idx])

                if actual_pkl_path and os.path.exists(actual_pkl_path):
                    with open(actual_pkl_path, 'rb') as f_pkl:
                        d = pickle.load(f_pkl)
                    trans = d['trans'].detach().cpu().numpy().flatten()
                    orient_aa = d['orient'].detach().cpu().numpy().flatten()
                    orient_mat = Rotation.from_rotvec(orient_aa).as_matrix()
                else:
                    valid_slice = False
                    break

                root_trans_list.append(trans)
                root_orient_list.append(orient_mat)

            if not valid_slice:
                continue

            with torch.no_grad():
                imgs_t = torch.stack(img_batch).to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    features = dinov2(imgs_t)
                all_fv.append(features.cpu().float())

            root_trans_arr = np.stack(root_trans_list)
            root_orient_arr = np.stack(root_orient_list)
            rel_motion = self.absolute_to_relative_motion(root_trans_arr, root_orient_arr)
            all_motion.append(torch.tensor(rel_motion, dtype=torch.float32))

        if len(all_fv) == 0:
            raise RuntimeError("error")

        torch.save(torch.stack(all_fv), os.path.join(self.cache_dir, "fv_full.pth"))
        torch.save(torch.stack(all_motion), os.path.join(self.cache_dir, "motion_full.pth"))


    def __getitem__(self, index):
        return {
            'fv_base': self.fv_full[index],
            'rel_motion': self.motion_full[index]
        }