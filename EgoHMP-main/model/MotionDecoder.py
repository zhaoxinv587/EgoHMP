import torch
import torch.nn as nn
import smplx
from human_body_prior.tools.model_loader import load_vposer
from gcn import GCN

class MotionDecoder(nn.Module):
    """
    Two-stage Graph Convolutional Network (GCN) for kinematic refinement.
    Integrates explicit geometric cues (SMPL-X) and physiological priors (VPoser),
    followed by absolute trajectory alignment to prevent root drift.
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = getattr(config, 'output_seq_len', 10)
        self.joint_num = 23
        self.vposer, _ = load_vposer('', vp_model='snapshot')
        self.vposer = self.vposer.to(self.device).eval()
        for param in self.vposer.parameters():
            param.requires_grad = False
        self.smplx = smplx.create('',
                                  model_type='smplx',
                                  gender='neutral',
                                  use_pca=False,
                                  ext='npz').to(self.device)
        self.smplx.eval()
        for param in self.smplx.parameters():
            param.requires_grad = False

        self.geom_dim = self.joint_num * 3  # 23 * 3 = 69
        self.latent_dim = 32  # VPoser latent code
        self.node_n = self.geom_dim + self.latent_dim

        hidden_dim = getattr(config, 'motion_hidden_dim', getattr(config, 'pose_hidden_dim', 256))
        p_dropout = getattr(config, 'dropout', 0.1)
        seq_len = self.input_seq_len + self.output_seq_len

        self.motion_gen = GCN(
            in_frames=self.input_seq_len,
            out_frames=self.output_seq_len,
            hidden_dim=hidden_dim,
            dropout=p_dropout,
            input_feature=seq_len,
            num_stage=9,
            node_n=self.node_n
        )

        self.motion_decoder = GCN(
            in_frames=self.input_seq_len,
            out_frames=self.output_seq_len,
            hidden_dim=hidden_dim,
            dropout=p_dropout,
            input_feature=seq_len,
            num_stage=9,
            node_n=self.node_n
        )

        self.output_proj = nn.Linear(self.node_n, self.geom_dim)

    def forward(self, phase3_out, hist_joints=None):
        B, T, _ = phase3_out.shape
        current_batch_size = B * T
        pred_traj = phase3_out[:, :, :6]  # [B, T, 6]
        pred_latent_pose = phase3_out[:, :, 6:]  # [B, T, 32]
        latent_flat = pred_latent_pose.reshape(current_batch_size, self.latent_dim)

        body_pose_rot = self.vposer.decode(latent_flat, output_type='aa')['pose_body']
        body_pose_rot = body_pose_rot.reshape(current_batch_size, 63)
        pred_traj_flat = pred_traj.reshape(current_batch_size, 6)
        global_orient = pred_traj_flat[:, :3]
        transl = pred_traj_flat[:, 3:6]

        kwargs = {
            'jaw_pose': self.smplx.jaw_pose.expand(current_batch_size, -1),
            'leye_pose': self.smplx.leye_pose.expand(current_batch_size, -1),
            'reye_pose': self.smplx.reye_pose.expand(current_batch_size, -1),
            'expression': self.smplx.expression.expand(current_batch_size, -1),
            'left_hand_pose': self.smplx.left_hand_pose.expand(current_batch_size, -1),
            'right_hand_pose': self.smplx.right_hand_pose.expand(current_batch_size, -1),
            'betas': self.smplx.betas.expand(current_batch_size, -1)
        }
        smplx_out = self.smplx(
            global_orient=global_orient,
            transl=transl,
            body_pose=body_pose_rot,
            **kwargs
        )

        j3d_smplx = smplx_out.joints[:, :self.joint_num, :]
        j3d_smplx = j3d_smplx.reshape(B, T, self.joint_num, 3)

        j3d_flat = j3d_smplx.reshape(B, T, self.geom_dim)
        gcn_input = torch.cat([j3d_flat, pred_latent_pose], dim=-1)

        refined_feats = self.motion_gen(gcn_input)
        refined_feats = self.motion_decoder(refined_feats)

        out_flat = self.output_proj(refined_feats)
        predictions = out_flat.reshape(B, T, self.joint_num, 3)

        root_target = pred_traj[:, :, 3:6].unsqueeze(-2)
        root_drift = predictions[:, :, [0], :] - root_target

        predictions = predictions - root_drift
        predictions[:, :, [0], :] = root_target

        if hist_joints is not None and hist_joints.shape[1] > 0:
            hist_len = hist_joints.shape[1]
            predictions[:, :hist_len, :, :] = hist_joints

        return predictions