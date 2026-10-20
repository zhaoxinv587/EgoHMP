import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
from human_body_prior.tools.model_loader import load_vposer
import smplx
from torch.cuda.amp import autocast, GradScaler
from dataset import EgoDataset
from model.diffusion import MotionDiffusion

def calc_traj_6d_loss(pred_traj, gt_traj, rot_weight=1.0):
    pred_rot = pred_traj[:, :, :3]
    pred_trans = pred_traj[:, :, 3:6]
    gt_rot = gt_traj[:, :, :3]
    gt_trans = gt_traj[:, :, 3:6]
    trans_error = torch.norm(pred_trans - gt_trans, dim=-1)
    rot_error = torch.norm(pred_rot - gt_rot, dim=-1)
    ade_trans = trans_error.mean()
    fde_trans = trans_error[:, -1].mean()
    ade_rot = rot_error.mean()
    total_loss = ade_trans + rot_weight * ade_rot
    return total_loss, ade_trans, fde_trans, ade_rot

def vposer_decode_safe(vposer, latent_code):
    rotmat = vposer.decode(latent_code).view(-1, 3, 3)
    tr = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    v = torch.stack([
        rotmat[:, 2, 1] - rotmat[:, 1, 2],
        rotmat[:, 0, 2] - rotmat[:, 2, 0],
        rotmat[:, 1, 0] - rotmat[:, 0, 1]
    ], dim=1)
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta + 1e-8)
    s = torch.where(theta < 1e-4, torch.full_like(theta, 0.5), theta / (2.0 * sin_theta))
    aa = v * s.unsqueeze(1)
    return aa.view(-1, 63)

class TrainConfig:
    def __init__(self):
        self.dataroot = './GIMO'
        self.p2_logits_dir = ''
        self.input_seq_len = 6
        self.fps = 2
        self.batch_size = 4
        self.accum_steps = 2
        self.num_workers = 4
        self.epochs = 121
        self.lr = 1e-4
        self.save_dir = "checkpoints/"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig()
    os.makedirs(config.save_dir, exist_ok=True)

    dataset = EgoDataset(config=config, train=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model = MotionDiffusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    scaler = GradScaler()

    body_model_decoder, _ = load_vposer('', vp_model='snapshot')
    body_model_decoder.eval().to(device)

    smplx_layer = smplx.create(model_path='', model_type='smplx', gender='neutral', ext='npz').to(device)
    smplx_layer.eval()

    for param in body_model_decoder.parameters(): param.requires_grad = False
    for param in smplx_layer.parameters(): param.requires_grad = False

    model.train()
    lambda_refined = 1.0
    lambda_coarse = 0.5
    lambda_contact = 0.1
    for epoch in range(config.epochs):
        epoch_refined_loss_sum = 0.0
        epoch_coarse_loss_sum = 0.0
        epoch_contact_loss_sum = 0.0
        epoch_mpjpe_path_sum = 0.0
        epoch_mpjpe_end_sum = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} [LR: {lr_scheduler.get_last_lr()[0]:.2e}]")
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            gt_motion = batch['gt_motion'].to(device, non_blocking=True)
            fv = batch['fv'].to(device, non_blocking=True)
            fm = batch['fm'].to(device, non_blocking=True)
            hist_traj = batch['hist_traj'].to(device, non_blocking=True)
            pts_local_full = batch['pts_local_full'].to(device, non_blocking=True)
            affordance_gt = batch["affordance_gt"]
            noise = torch.randn_like(gt_motion)
            timesteps = torch.randint(0, 1000, (gt_motion.shape[0],), device=device)
            x_noisy = noise_scheduler.add_noise(gt_motion, noise, timesteps)

            with autocast():
                outputs = model(x_noisy, timesteps, fv, fm, pts_local_full, hist_traj)
                out_refined = outputs["phase3_out"]
                out_coarse = outputs["coarse_out"]
                contact_prob = outputs["contact_prob"]
                loss_refined = F.mse_loss(out_refined[:, :, 6:], gt_motion[:, :, 6:])
                loss_coarse = F.mse_loss(out_coarse[:, :, 6:], gt_motion[:, :, 6:])
                gt_dist = affordance_gt / (affordance_gt.sum(dim=-1, keepdim=True) + 1e-8)
                pred_dist = contact_prob / (contact_prob.sum(dim=-1, keepdim=True) + 1e-8)
                pred_log_dist = torch.log(pred_dist + 1e-8)
                loss_contact = F.kl_div(pred_log_dist, gt_dist, reduction='batchmean')
                total_loss = (lambda_refined * loss_refined +
                              lambda_coarse * loss_coarse +
                              lambda_contact * loss_contact)
                total_loss = total_loss / config.accum_steps


            scaler.scale(total_loss).backward()
            if (num_batches + 1) % config.accum_steps == 0 or (num_batches + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                B, T, _ = out_refined.shape
                pred_orient = out_refined[:, :, 0:3].reshape(B * T, 3)
                pred_trans = out_refined[:, :, 3:6].reshape(B * T, 3)
                pred_latent = out_refined[:, :, 6:].reshape(B * T, 32)
                pred_body_pose = vposer_decode_safe(body_model_decoder, pred_latent)
                kwargs = {
                    'jaw_pose': smplx_layer.jaw_pose.expand(B * T, -1),
                    'leye_pose': smplx_layer.leye_pose.expand(B * T, -1),
                    'reye_pose': smplx_layer.reye_pose.expand(B * T, -1),
                    'expression': smplx_layer.expression.expand(B * T, -1),
                    'left_hand_pose': smplx_layer.left_hand_pose.expand(B * T, -1),
                    'right_hand_pose': smplx_layer.right_hand_pose.expand(B * T, -1),
                    'betas': smplx_layer.betas.expand(B * T, -1)
                }
                pred_output = smplx_layer(global_orient=pred_orient, body_pose=pred_body_pose, transl=pred_trans,
                                          **kwargs)
                pred_joints = pred_output.joints.reshape(B, T, -1, 3)
                gt_orient = gt_motion[:, :, 0:3].reshape(B * T, 3)
                gt_trans = gt_motion[:, :, 3:6].reshape(B * T, 3)
                gt_latent = gt_motion[:, :, 6:].reshape(B * T, 32)
                gt_body_pose = vposer_decode_safe(body_model_decoder, gt_latent)
                gt_output = smplx_layer(global_orient=gt_orient, body_pose=gt_body_pose, transl=gt_trans, **kwargs)
                gt_joints = gt_output.joints.reshape(B, T, -1, 3)
                joint_diff = torch.norm(pred_joints[:, :, :22, :] - gt_joints[:, :, :22, :], dim=-1)
                mpjpe_path_batch = joint_diff.mean().item()
                mpjpe_end_batch = joint_diff[:, -1, :].mean().item()

            epoch_refined_loss_sum += loss_refined.item()
            epoch_coarse_loss_sum += loss_coarse.item()
            epoch_contact_loss_sum += loss_contact.item()
            epoch_mpjpe_path_sum += mpjpe_path_batch
            epoch_mpjpe_end_sum += mpjpe_end_batch
            num_batches += 1
            pbar.set_postfix({
                'L_Ref': f"{loss_refined.item():.3f}",
                'L_Coa': f"{loss_coarse.item():.3f}",
                'L_KL': f"{loss_contact.item():.3f}",
                'MPJPE': f"{mpjpe_path_batch:.3f}",
            })

        lr_scheduler.step()
        avg_ref = epoch_refined_loss_sum / num_batches
        avg_coa = epoch_coarse_loss_sum / num_batches
        avg_kl = epoch_contact_loss_sum / num_batches
        avg_mpjpe_path = epoch_mpjpe_path_sum / num_batches
        avg_mpjpe_end = epoch_mpjpe_end_sum / num_batches

        print(f"\n  Epoch [{epoch}/{config.epochs}] Summary:")
        print(f"   [Loss] Refined: {avg_ref:.4f} | Coarse: {avg_coa:.4f} | Contact KL: {avg_kl:.4f}")
        print(f"   [Eval] MPJPE_Path: {avg_mpjpe_path:.4f} m | MPJPE_End: {avg_mpjpe_end:.4f} m\n")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    train()
