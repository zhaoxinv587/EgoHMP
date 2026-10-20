import torch
import torch.nn as nn
from pointnet_plus2 import PointNet2SemSegSSGShape

class IntentTransformer(nn.Module):

    def __init__(self, v_dim=1024, m_dim=1024, embed_dim=768, nhead=8, num_layers=2):
        super().__init__()
        self.v_proj = nn.Sequential(
            nn.Linear(v_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, embed_dim)
        )

        self.m_proj = nn.Sequential(
            nn.Linear(m_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, embed_dim)
        )

        self.v_pos_embedding = nn.Parameter(torch.randn(1, 6, embed_dim))
        self.m_pos_embedding = nn.Parameter(torch.randn(1, 6, embed_dim))

        self.v_mod_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.m_mod_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.intent_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fv, fm):
        batch_size = fv.size(0)

        feat_v = self.v_proj(fv)  # [B, 6, 256]
        feat_m = self.m_proj(fm)  # [B, 6, 256]

        feat_v = feat_v + self.v_pos_embedding + self.v_mod_token
        feat_m = feat_m + self.m_pos_embedding + self.m_mod_token

        tokens = torch.cat([feat_v, feat_m], dim=1)  # [B, 12, 256]

        intent_tokens = self.intent_token.expand(batch_size, -1, -1)
        full_tokens = torch.cat([intent_tokens, tokens], dim=1)  # [B, 13, 256]

        output = self.transformer(full_tokens)

        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query_pts, key_intent):
        # query_pts: [B, N, dim], key_intent: [B, 1, dim]
        attn_out, _ = self.attn(query_pts, key_intent, key_intent)
        x = self.norm(query_pts + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x




class EgoNet(nn.Module):
    def __init__(self, p_dim=768, n_attn_layers=3, up_axis=2, height_ratio=0.5):
        super().__init__()
        self.p_dim = p_dim
        self.up_axis = up_axis
        self.height_ratio = height_ratio

        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': p_dim})

        self.temporal_pooler = IntentTransformer(
            v_dim=1024, m_dim=1024, embed_dim=p_dim, nhead=8, num_layers=2
        )

        self.attention_layers = nn.ModuleList([
            CrossAttentionBlock(dim=p_dim) for _ in range(n_attn_layers)
        ])

        self.heatmap_head = nn.Sequential(
            nn.Conv1d(p_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, fv, fm, points_xyz_rgb):
        key_intent = self.temporal_pooler(fv, fm)
        pts_input = points_xyz_rgb.transpose(1, 2)
        local_p, _ = self.pointnet(pts_input)
        query_pts = local_p.transpose(1, 2)  # [B, N, p_dim]
        x = query_pts
        for layer in self.attention_layers:
            x = layer(x, key_intent)  # [B, N, p_dim]

        global_intent = key_intent[:, 0, :].unsqueeze(1)  # [B, 1, p_dim]
        similarity_score = (x * global_intent).sum(dim=-1)  # [B, N]
        similarity_score = similarity_score / (self.p_dim ** 0.5)
        fusion_feat = x.transpose(1, 2)  # [B, p_dim, N]
        mlp_score = self.heatmap_head(fusion_feat).squeeze(1)  # [B, N]
        logits = mlp_score + similarity_score

        return logits, fusion_feat