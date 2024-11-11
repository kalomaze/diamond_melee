from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import numpy as np
from PIL import Image
import time

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    upsampling_factor: Optional[int] = None


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        cfg.inner_model.is_upsampler = self.is_upsampler
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None
        
        # Setup output directories
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_out_dir = Path("images_out") / timestamp
        self.gifs_dir = self.base_out_dir / "gifs"
        self.context_dir = self.base_out_dir / "context"
        
        # Create directories
        self.base_out_dir.mkdir(exist_ok=True, parents=True)
        self.gifs_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Optional[Tensor], cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], obs: Tensor, act: Optional[Tensor]) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def tensor_to_numpy_img(self, x: Tensor) -> np.ndarray:
        """Convert tensor to numpy image in uint8 format"""
        # Convert from [-1,1] to [0,1] range
        x = (x + 1) / 2
        x = (x.cpu().numpy() * 255).astype(np.uint8)

        if len(x.shape) == 4:  # BCHW format
            if x.shape[1] == 3:
                # Single RGB frame
                x = np.transpose(x, (0,2,3,1))[...,::-1]  # BCHW -> BHWC, BGR -> RGB
            else:
                # Multiple frames concatenated in channel dimension
                frames = x.shape[1] // 3  # Calculate number of frames
                x = x.reshape(x.shape[0], frames, 3, x.shape[2], x.shape[3])  # B(NC)HW -> BNCHW
                x = np.transpose(x, (0,1,3,4,2))[...,::-1]  # BNCHW -> BNHWC, BGR -> RGB
        elif len(x.shape) == 5:  # BNCHW format
            x = np.transpose(x, (0,1,3,4,2))[...,::-1]  # BNCHW -> BNHWC, BGR -> RGB

        return x

    def save_context_grid(self, conditioning: np.ndarray, unique_id: str, batch_idx: int) -> None:
        """Save conditioning frames as a single grid image"""
        frames_per_row = self.cfg.inner_model.num_steps_conditioning
        h, w = conditioning.shape[1:3]  # Get height and width from the input
        
        # Create a grid image
        grid = Image.new('RGB', (w * frames_per_row, h))
        
        # Paste each frame into the grid
        for i in range(frames_per_row):
            frame = Image.fromarray(conditioning[i])
            grid.paste(frame, (i * w, 0))
        
        grid.save(self.context_dir / f"{unique_id}_batch{batch_idx}_context_grid.png")

    def save_denoising_gif(self, frames: List[Image.Image], target: Image.Image, unique_id: str) -> None:
        """
        Save a GIF showing the denoising process followed by alternating comparison
        between final output and target.
        """
        # Denoising progression frames
        output_frames = frames.copy()
        final_denoised = frames[-1]
        
        # Add alternating comparisons at the end
        for _ in range(2):
            output_frames.append(final_denoised)
            output_frames.append(target)
        
        # Set frame durations - faster for progression, slower for comparison
        durations = [200] * len(frames)  # 200ms for progression
        durations.extend([500] * 4)  # 500ms for comparison frames
        
        # Save the GIF
        output_frames[0].save(
            self.gifs_dir / f"{unique_id}_denoising.gif",
            save_all=True,
            append_images=output_frames[1:],
            duration=durations,
            loop=0
        )

    def forward(self, batch: Batch) -> LossAndLogs:
        b, t, c, h, w = batch.obs.size()
        H, W = (self.cfg.upsampling_factor * h, self.cfg.upsampling_factor * w) if self.is_upsampler else (h, w)
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n

        if self.is_upsampler:
            all_obs = torch.stack([x["full_res"] for x in batch.info]).to(self.device)
            low_res = F.interpolate(batch.obs.reshape(b * t, c, h, w), scale_factor=self.cfg.upsampling_factor, mode="bicubic").reshape(b, t, c, H, W)
            assert all_obs.shape == low_res.shape
        else:
            all_obs = batch.obs.clone()

        loss = 0
        frames_to_save = []
        should_save = False
        last_prev_obs = None

        for i in range(seq_length):
            prev_obs = all_obs[:, i : n + i].reshape(b, n * c, H, W)
            prev_act = None if self.is_upsampler else batch.act[:, i : n + i]
            obs = all_obs[:, n + i]
            mask = batch.mask_padding[:, n + i]

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, self.device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            if self.is_upsampler:
                prev_obs = torch.cat((prev_obs, low_res[:, n + i]), dim=1)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(noisy_obs, prev_obs, prev_act, cs)

            target = (obs - cs.c_skip * noisy_obs) / cs.c_out
            loss += F.mse_loss(model_output[mask], target[mask])

            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            all_obs[:, n + i] = denoised

            if i == seq_length - 1 and torch.rand(1).item() < 0.02:
                should_save = True
                last_prev_obs = prev_obs.clone()
                frames_to_save = [all_obs[:, n + j].clone() for j in range(seq_length)]

        loss /= seq_length

        if should_save and last_prev_obs is not None:
            timestamp = int(time.time())
            unique_id = f"epoch{getattr(batch, 'epoch', 0)}_step{timestamp}"
            
            # Save context frames as grid
            cond_np = self.tensor_to_numpy_img(last_prev_obs)[0]  # Get first batch
            self.save_context_grid(cond_np, unique_id, 0)
            
            # Create and save denoising GIF with comparisons
            frames = [Image.fromarray(self.tensor_to_numpy_img(f)[0]) for f in frames_to_save]
            target = Image.fromarray(self.tensor_to_numpy_img(batch.obs[:, -1])[0])
            self.save_denoising_gif(frames, target, unique_id)

        return loss, {"loss_denoising": loss.item()}