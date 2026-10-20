import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from VMAE import VisualEncoder, MotionEncoder, alignment_loss
from dataset import EgoMotionPretrainDataset


class DummyConfig:
    def __init__(self):
        self.dataroot = './GIMO'
        self.input_seq_len = 6
        self.fps = 2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DummyConfig()
    train_dataset = EgoMotionPretrainDataset(config, train=True)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    motion_encoder = MotionEncoder().to(device)
    visual_st_encoder = VisualEncoder().to(device)

    trainable_params = list(motion_encoder.parameters()) + list(visual_st_encoder.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    num_epochs = 80

    motion_encoder.train()
    visual_st_encoder.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_nce = 0.0
        epoch_acc = 0.0
        epoch_std = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            fv_base = batch['fv_base'].to(device)  # [B, 6, 1024]
            rel_motion = batch['rel_motion'].to(device)  # [B, 6, 12]
            B = fv_base.shape[0]

            fv_temporal, fv_proj = visual_st_encoder(fv_base)
            fm_temporal, fm_proj = motion_encoder(rel_motion)

            loss, l_kl, l_nce = alignment_loss(
                fm_temporal, fv_temporal,
                fm_proj, fv_proj,
                lambda_contrast=5.0,
                temp_kl=1.0
            )

            with torch.no_grad():
                fm_norm = F.normalize(fm_proj, p=2, dim=-1)
                fv_norm = F.normalize(fv_proj, p=2, dim=-1)
                sim_matrix = torch.matmul(fm_norm, fv_norm.transpose(0, 1))

                preds = torch.argmax(sim_matrix, dim=1)
                labels = torch.arange(B).to(device)
                correct = (preds == labels).sum().item()
                retrieval_acc = correct / B

                feat_std = fm_proj.std(dim=0).mean().item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            epoch_kl += l_kl.item()
            epoch_nce += l_nce.item()
            epoch_acc += retrieval_acc
            epoch_std += feat_std


            progress_bar.set_postfix({
                'L_KL': f"{l_kl.item():.2f}",
                'L_NCE': f"{l_nce.item():.2f}",
                'Acc': f"{retrieval_acc:.1%}",
                'Std': f"{feat_std:.3f}"
            })


        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_kl = epoch_kl / num_batches
        avg_nce = epoch_nce / num_batches
        avg_acc = epoch_acc / num_batches
        avg_std = epoch_std / num_batches

        print(
            f" Epoch {epoch + 1} | Avg L_KL: {avg_kl:.4f} | Avg L_NCE: {avg_nce:.4f} | Acc: {avg_acc:.2%} | Std: {avg_std:.4f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'motion_encoder': motion_encoder.state_dict(),
                'visual_st_encoder': visual_st_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, f"checkpoints/phase1_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    main()