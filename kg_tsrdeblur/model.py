"""Sections 2.1--2.2, equations (6)--(9). All unresolved choices: docs/IMPLEMENTATION.md."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 32
    encoder_depths: tuple = (2, 2)
    decoder_depths: tuple = (2, 2)
    transformer_blocks: int = 4
    heads: int = 4
    window: int = 8
    motion_dim: int = 32
    refinement_channels: int = 16
    image_only: bool = False
    spatial_only: bool = False
    conv_bottleneck: bool = False


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.local = nn.Sequential(nn.Conv2d(c, 2*c, 1),
            nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c), SimpleGate())
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.out = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(nn.Conv2d(c, 2*c, 1), SimpleGate(), nn.Conv2d(c, c, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.local(self.norm1(x))
        x = x + self.beta * self.out(y * self.sca(y))
        return x + self.gamma * self.ffn(self.norm2(x))


class MotionAdapter(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.project = nn.Linear(d, c)
        self.norm = LayerNorm2d(c)
        self.gate = nn.Conv2d(2*c, 1, 1)
        self.correction = nn.Sequential(nn.Conv2d(2*c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, x, descriptor, reliability):
        v = self.project(self.encoder(descriptor))[:, :, None, None].expand_as(x)
        joint = torch.cat([self.norm(x), v], dim=1)
        gate = reliability.reshape(-1, 1, 1, 1) * self.gate(joint).sigmoid()
        return x + gate * self.correction(joint), gate


class WindowBlock(nn.Module):
    """Alternating window attention with cyclic-shift and padding masks."""
    def __init__(self, c, heads, window, shift=0):
        super().__init__()
        if c % heads:
            raise ValueError("Bottleneck channels must be divisible by heads")
        self.window, self.shift, self.heads = window, shift, heads
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.qkv, self.proj = nn.Linear(c, 3*c), nn.Linear(c, c)
        self.ffn = nn.Sequential(nn.Conv2d(c, 2*c, 1),
            nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c), nn.GELU(), nn.Conv2d(2*c, c, 1))

    def partition(self, x):
        b, c, h, w = x.shape
        s = self.window
        return x.reshape(b, c, h//s, s, w//s, s).permute(0, 2, 4, 3, 5, 1).reshape(-1, s*s, c)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.window
        ph, pw = (-h) % s, (-w) % s
        hp, wp = h+ph, w+pw
        shift = self.shift if min(hp, wp) > s else 0
        y = F.pad(self.norm1(x), (0, pw, 0, ph))
        valid = torch.zeros(1, 1, hp, wp, device=x.device)
        valid[:, :, :h, :w] = 1
        if shift:
            y = torch.roll(y, (-shift, -shift), (-2, -1))
            valid = torch.roll(valid, (-shift, -shift), (-2, -1))
        tokens = self.partition(y)
        n = s*s
        qkv = self.qkv(tokens).reshape(-1, n, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        # Region IDs block wrap-around interactions introduced by a cyclic shift.
        regions = torch.zeros(1, 1, hp, wp, device=x.device)
        if shift:
            idx = 0
            for hs in (slice(0, -s), slice(-s, -shift), slice(-shift, None)):
                for ws in (slice(0, -s), slice(-s, -shift), slice(-shift, None)):
                    regions[:, :, hs, ws] = idx
                    idx += 1
        ids = self.partition(regions).squeeze(-1)
        good = self.partition(valid).squeeze(-1).bool()
        allowed = (ids[:, :, None] == ids[:, None, :]) & good[:, None, :]
        # Padded queries can self-attend; discarded after attention, avoiding all-masked rows.
        eye = torch.eye(n, device=x.device, dtype=torch.bool)[None]
        allowed = allowed | ((~good)[:, :, None] & eye)
        mask = allowed.repeat(b, 1, 1)[:, None]
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        out = self.proj(out.transpose(1, 2).reshape(-1, n, c))
        y = out.reshape(b, hp//s, wp//s, s, s, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        if shift:
            y = torch.roll(y, (shift, shift), (-2, -1))
        x = x + y[:, :, :h, :w]
        return x + self.ffn(self.norm2(x))


class SpatialFrequencyRefinement(nn.Module):
    def __init__(self, c, reduced, spatial_only=False):
        super().__init__()
        self.spatial_only = spatial_only
        self.reduce = nn.Conv2d(c, reduced, 1)
        self.spatial = nn.Sequential(nn.Conv2d(reduced, reduced, 3, padding=1, groups=reduced), nn.Conv2d(reduced, reduced, 1))
        self.spectral = nn.Sequential(nn.Conv2d(reduced, reduced, 1), nn.GELU(), nn.Conv2d(reduced, reduced, 1))
        self.fusion = nn.Conv2d(2*reduced, 1, 1)
        self.project = nn.Conv2d(reduced, c, 1)

    def forward(self, x):
        u = self.reduce(x)
        spa = self.spatial(u)
        if self.spatial_only:
            return x + self.project(spa)
        # FFT in float32 supports arbitrary ROI sizes even under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = torch.fft.rfft2(u.float(), norm="ortho")
            gain = 2 * self.spectral(torch.log1p(z.abs())).sigmoid()
            freq = torch.fft.irfft2(gain*z, s=u.shape[-2:], norm="ortho")
        freq = freq.to(spa.dtype)
        weight = self.fusion(torch.cat([spa, freq], dim=1)).sigmoid()
        return x + self.project(weight*spa + (1-weight)*freq)


class KGTSRDeblur(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg, c = self.config, self.config.width
        if len(cfg.encoder_depths) != 2 or len(cfg.decoder_depths) != 2:
            raise ValueError("Exactly two encoder and decoder stages required")
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        for depth in cfg.encoder_depths:
            self.encoders.append(nn.Sequential(*(NAFBlock(c) for _ in range(depth))))
            self.downs.append(nn.Conv2d(c, 2*c, 2, stride=2))
            c *= 2
        self.motion = MotionAdapter(c, cfg.motion_dim)
        self.bottleneck = nn.Sequential(*(
            NAFBlock(c) if cfg.conv_bottleneck else WindowBlock(c, cfg.heads, cfg.window, (i%2)*(cfg.window//2))
            for i in range(cfg.transformer_blocks)))
        self.ups, self.fusions, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for depth in cfg.decoder_depths:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.PixelShuffle(2)))
            self.fusions.append(nn.Conv2d(c, c//2, 1))
            c //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(c) for _ in range(depth))))
        self.refine = SpatialFrequencyRefinement(c, cfg.refinement_channels, cfg.spatial_only)
        self.head = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, image, descriptor=None, reliability=None, return_gate=False):
        b, _, h, w = image.shape
        if descriptor is None:
            descriptor = image.new_zeros(b, 16)
        if reliability is None or self.config.image_only:
            reliability = image.new_zeros(b)
        if descriptor.shape != (b, 16) or reliability.numel() != b:
            raise ValueError("Expected descriptor [B,16] and reliability [B]")
        if not torch.isfinite(descriptor).all() or not torch.isfinite(reliability).all():
            raise ValueError("Motion inputs must be finite")
        if ((reliability < 0) | (reliability > 1)).any():
            raise ValueError("Reliability must be in [0,1]")
        x = self.stem(F.pad(image, (0, (-w)%4, 0, (-h)%4), mode="replicate"))
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x, gate = self.motion(x, descriptor, reliability)
        x = self.bottleneck(x)
        for up, fuse, dec, skip in zip(self.ups, self.fusions, self.decoders, reversed(skips)):
            x = dec(fuse(torch.cat([up(x), skip], dim=1)))
        restored = image + self.head(self.refine(x))[:, :, :h, :w]
        return (restored, gate) if return_gate else restored
