import torch
import torch.nn as nn
import numpy as np
from base_cross_model import CrossAttentionLayer
from IntentNet import EgoNet

def aa_to_rotmat(axis_angle):
    B = axis_angle.shape[0]
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    K = torch.zeros(B, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0)
    outer_prod = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))
    rot_mat = cos_a * I + sin_a * K + (1 - cos_a) * outer_prod
    return rot_mat

def load_smplx_skinning_weights(model_path='', num_body_parts=22):
    try:
        import smplx
        smplx_model = smplx.create(model_path=model_path, model_type='smplx', gender='neutral', ext='npz')
        lbs_weights = smplx_model.lbs_weights.detach().clone()
        if num_body_parts < 55:
            lbs_weights = lbs_weights[:, :num_body_parts]
            lbs_weights = lbs_weights / (lbs_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return lbs_weights
    except Exception as e:
        fallback_weights = torch.ones(10475, num_body_parts) / num_body_parts
        return fallback_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe[None])

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_enc):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, timesteps):
        t = self.pos_enc.pe[0, timesteps]
        return self.time_embed(t)


class MotionDiffusion(nn.Module):
    def __init__(self, smplx_model_path=''):
        super().__init__()

        self.latent_dim = 768
        self.input_feats = 38
        self.fv_dim = 1024
        self.fm_dim = 1024
        self.dimension = 768
        self.input_seq_len = 6
        self.output_seq_len = 10
        self.seq_len = self.input_seq_len + self.output_seq_len
        self.num_layers = 12
        self.num_heads = 12
        self.dropout = 0.1
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_fv = nn.Linear(self.fv_dim, self.latent_dim)
        self.embed_fm = nn.Linear(self.fm_dim, self.latent_dim)
        from model.core import DecoderBlock
        self.tsfm = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.num_heads, self.dropout, 2)
            for _ in range(self.num_layers)
        ])


        self.scene_gate = nn.Sequential(
            nn.Linear(self.dimension, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        self.past_traj_encoder = nn.Linear(6, self.latent_dim)
        self.goal_encoder = nn.Linear(3, self.latent_dim)

        self.ts_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.latent_dim, num_kv_channels=self.dimension, num_heads=8)
            for _ in range(8)
        ])
        self.fm_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.latent_dim, num_kv_channels=self.dimension, num_heads=8)
            for _ in range(8)
        ])

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, batch_first=True)
        self.traj_encoder_refiner = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.traj_input_mlp = nn.Linear(self.latent_dim, self.latent_dim)
        self.traj_output_mlp = nn.Linear(self.latent_dim, 6)

        self.num_local_tokens = 4
        self.local_queries = nn.Parameter(torch.randn(1, self.num_local_tokens, self.latent_dim))
        self.local_attn_pool = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=8, batch_first=True
        )
        self.coarse_pose_head = nn.Linear(self.latent_dim, 32)
        self.contact_head = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 10475),
            nn.Sigmoid()
        )

        self.num_body_parts = 22
        official_weights = load_smplx_skinning_weights(smplx_model_path, num_body_parts=self.num_body_parts)
        self.register_buffer('kinematic_pooling_weights', official_weights)

        self.contact_feat_encoder = nn.Sequential(
            nn.Linear(self.num_body_parts, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.adaln_generator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, self.latent_dim * 2)
        )

        nn.init.constant_(self.adaln_generator[-1].weight, 0)
        nn.init.constant_(self.adaln_generator[-1].bias, 0)

        refiner_in_dim = self.latent_dim + 32 + 6
        self.global_refiner = nn.Sequential(
            nn.Linear(refiner_in_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 32)
        )

        nn.init.constant_(self.global_refiner[-1].weight, 0)
        nn.init.constant_(self.global_refiner[-1].bias, 0)
        self.modelintent = EgoNet(p_dim=self.dimension)
    def forward(self, x_noisy, timesteps, fv, fm, pts_local_full, hist_traj):
        B, T, _ = x_noisy.shape
        logits, fusion_feat= self.modelintent(fv, fm, pts_local_full)

        best_idx = torch.argmax(logits, dim=-1)
        batch_indices = torch.arange(B, device=pts_local_full.device)
        interaction_location = pts_local_full[batch_indices, best_idx, :3]

        dist = torch.norm(pts_local_full[..., :3] - interaction_location.unsqueeze(1), dim=-1, keepdim=True)
        dist_min, dist_max = dist.min(dim=1, keepdim=True)[0], dist.max(dim=1, keepdim=True)[0]
        distance_map = (dist - dist_min) / (dist_max - dist_min + 1e-8)
        distance_map = 1 - distance_map

        scene_feats = fusion_feat * distance_map
        scene_feats = self.scene_gate(scene_feats)

        F_past = self.past_traj_encoder(hist_traj)  # [B, 6, 768]
        F_goal = self.goal_encoder(interaction_location).unsqueeze(1)  # [B, 1, 768]
        F_current = F_past[:, -1:, :]  # [B, 1, 768]
        F_middle = F_current.repeat(1, self.output_seq_len - 1, 1)  # [B, 9, 768]
        F_traj = torch.cat([F_past, F_middle, F_goal], dim=1)  # [B, 16, 768]

        for layer in self.ts_cross_att:
            F_traj = layer(F_traj, scene_feats)

        H = pts_local_full[..., [1]]
        phi, _ = torch.max(H, dim=1, keepdim=True)
        phi /= 2
        M = (H > phi).float()
        F_SM = scene_feats * (1 - M)
        for layer in self.fm_cross_att:
            F_traj = layer(F_traj, F_SM)
        F_traj_future_refined = F_traj[:, self.input_seq_len:, :]
        F_STS = torch.cat([F_past, F_traj_future_refined], dim=1)
        Traj_embed = self.traj_input_mlp(F_STS)
        Traj_embed = self.pos_enc(Traj_embed)
        pred_traj_embed = self.traj_encoder_refiner(Traj_embed)

        pred_traj = self.traj_output_mlp(pred_traj_embed)  # [B, 16, 6]
        traj_for_diffusion = pred_traj[:, self.input_seq_len:, :].detach()

        pos = traj_for_diffusion[:, :, 3:6]
        aa = traj_for_diffusion[:, :, 0:3]

        pts_xyz = pts_local_full[..., :3]
        pts_rel = pts_xyz.unsqueeze(1) - pos.unsqueeze(2)

        pts_rel_flat = pts_rel.view(B * T, -1, 3)
        rot_mat = aa_to_rotmat(aa.reshape(B * T, 3))
        pts_rel_rot = torch.bmm(pts_rel_flat, rot_mat)

        h = pts_rel_rot[:, :, 1]
        r2 = pts_rel_rot[:, :, 0] ** 2 + pts_rel_rot[:, :, 2] ** 2

        radius = 1.0
        h_min = -1.2
        h_max = 1.0
        cylinder_mask = (h >= h_min) & (h <= h_max) & (r2 <= radius ** 2)

        valid_counts = cylinder_mask.sum(dim=1)
        cylinder_mask[valid_counts == 0, 0] = True
        key_padding_mask = ~cylinder_mask

        kv_flat = fusion_feat.repeat_interleave(T, dim=0)
        q_flat = self.local_queries.repeat(B * T, 1, 1)

        f_local_detailed, _ = self.local_attn_pool(
            query=q_flat, key=kv_flat, value=kv_flat, key_padding_mask=key_padding_mask
        )
        local_feats_seq = f_local_detailed.reshape(B, T * self.num_local_tokens, self.latent_dim)

        x_full = torch.cat([traj_for_diffusion, x_noisy[:, :, 6:]], dim=-1)
        x_emb = self.input_process(x_full)
        x_emb = self.pos_enc(x_emb)
        enc_time = self.embed_timestep(timesteps)
        x_emb = torch.cat([enc_time.unsqueeze(1), x_emb], dim=1)

        enc_fv = self.embed_fv(fv)
        enc_fm = self.embed_fm(fm)

        context = torch.cat([enc_fv, enc_fm, local_feats_seq], dim=1)

        for dec in self.tsfm:
            x_emb = dec(x=x_emb, context=context)

        latent_feats = x_emb[:, 1:]  # [B, T, 768]


        coarse_latent = self.coarse_pose_head(latent_feats)  # [B, T, 32]
        contact_prob = self.contact_head(latent_feats)       # [B, T, 10475]

        part_contact_prob = torch.matmul(contact_prob, self.kinematic_pooling_weights)

        contact_feat = self.contact_feat_encoder(part_contact_prob)  # [B, T, 128]

        adaln_params = self.adaln_generator(contact_feat)    # [B, T, 1536]
        gamma, beta = adaln_params.chunk(2, dim=-1)          # [B, T, 768] each

        normed_latent = self.latent_norm(latent_feats)
        # H_mod = LayerNorm(H) * (1 + gamma) + beta
        modulated_latent_feats = normed_latent * (1.0 + gamma) + beta


        refine_input = torch.cat([
            modulated_latent_feats,
            coarse_latent.detach(),
            traj_for_diffusion
        ], dim=-1)

        delta_latent = self.global_refiner(refine_input)     # [B, T, 32]

        refined_latent = coarse_latent + delta_latent

        out_coarse = torch.cat([traj_for_diffusion, coarse_latent], dim=-1)
        out_refined = torch.cat([traj_for_diffusion, refined_latent], dim=-1)

        return {
            "phase3_out": out_refined,
            "coarse_out": out_coarse,
            "contact_prob": contact_prob,
        }