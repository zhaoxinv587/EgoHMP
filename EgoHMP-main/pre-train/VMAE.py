import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=1024, seq_len=6, num_heads=8):
        super().__init__()
        self.temporal_pe = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, batch_first=True, activation='gelu'
        )
        self.space_time_attention = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = x + self.temporal_pe

        fv_temporal = self.space_time_attention(x)  # [Batch, 6, 1024]
        fv_global = fv_temporal.mean(dim=1)  # [Batch, 1024]
        fv_proj = self.contrastive_head(fv_global)  # [Batch, 256]

        return fv_temporal, fv_proj


class MotionEncoder(nn.Module):

    def __init__(self, input_dim=12, hidden_dim=512, output_dim=1024, seq_len=6, proj_dim=256):
        super().__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.temporal_pe = nn.Parameter(torch.randn(1, seq_len, output_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=4,
            dim_feedforward=output_dim * 2,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.contrastive_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):

        fm_spatial = self.spatial_mlp(x)  # [Batch, 6, 1024]

        fm_pe = fm_spatial + self.temporal_pe

        fm_temporal = self.temporal_encoder(fm_pe)  # [Batch, 6, 1024]

        fm_global = fm_temporal.mean(dim=1)  # [Batch, 1024]

        fm_proj = self.contrastive_head(fm_global)  # [Batch, 256]

        return fm_temporal, fm_proj


def alignment_loss(fm_temporal, fv_temporal, fm_proj, fv_proj, lambda_contrast=5.0, temp_kl=1.0, temp_nce=0.07, eps=1e-8):

    B = fm_temporal.shape[0]
    device = fm_temporal.device

    fm = (fm_temporal - fm_temporal.mean(dim=-1, keepdim=True)) / (fm_temporal.std(dim=-1, keepdim=True) + eps)
    fv = (fv_temporal - fv_temporal.mean(dim=-1, keepdim=True)) / (fv_temporal.std(dim=-1, keepdim=True) + eps)

    i, j = 0, 5
    fm_i, fm_j = fm[:, i, :], fm[:, j, :]
    fv_i, fv_j = fv[:, i, :], fv[:, j, :]

    log_p_m_i = F.log_softmax(fm_i / temp_kl, dim=-1)
    log_p_m_j = F.log_softmax(fm_j / temp_kl, dim=-1)
    p_m_i = F.softmax(fm_i / temp_kl, dim=-1)

    log_p_v_i = F.log_softmax(fv_i / temp_kl, dim=-1)
    log_p_v_j = F.log_softmax(fv_j / temp_kl, dim=-1)
    p_v_i = F.softmax(fv_i / temp_kl, dim=-1)

    kl_m = torch.sum(p_m_i * (log_p_m_i - log_p_m_j), dim=-1)
    kl_v = torch.sum(p_v_i * (log_p_v_i - log_p_v_j), dim=-1)

    loss_kl = F.mse_loss(kl_m, kl_v)

    fm_norm = F.normalize(fm_proj, p=2, dim=-1)
    fv_norm = F.normalize(fv_proj, p=2, dim=-1)

    logits = torch.matmul(fm_norm, fv_norm.transpose(0, 1)) / temp_nce
    labels = torch.arange(B).to(device)
    loss_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


    total_loss = loss_kl + lambda_contrast * loss_nce

    return total_loss, loss_kl, loss_nce

