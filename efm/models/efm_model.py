# efm/models/efm_model.py
# Dual-branch Epilepsy Foundation Model (EFM):
# - Raw 1D branch (Temporal Transformer)
# - Time–Frequency branch (log-STFT -> 2D ViT-like encoder)
# - Gated cross-modal fusion
# - Heads: SSL (optional), Forecast (binary), Hazard (optional)

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Small building blocks
# ----------------------------

class ConvPatchEmbed1D(nn.Module):
    """Patchify time series into tokens via strided 1D conv."""

    def __init__(self, in_ch: int = 19, dim: int = 256, patch_len: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, dim, kernel_size=patch_len, stride=patch_len, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, N_t, D)
        z = self.proj(x)  # (B, D, N_t)
        return z.transpose(1, 2)  # (B, N_t, D)


class GatedFFN(nn.Module):
    """Gated MLP (residual)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        f = self.fc2(self.drop(self.act(self.fc1(x))))
        return x + g * f


class RelPosMHSA(nn.Module):
    """MultiheadAttention placeholder with room for relative bias (kept simple)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class RawTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosMHSA(dim, num_heads, dropout=drop)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim, dropout=drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class RawBranch(nn.Module):
    """1D temporal transformer on raw EEG."""

    def __init__(
            self,
            in_ch: int = 19,
            dim: int = 256,
            depth: int = 6,
            patch_len: int = 64,
            heads: int = 4,
            drop: float = 0.1,
    ):
        super().__init__()
        self.patch = ConvPatchEmbed1D(in_ch, dim, patch_len)
        self.blocks = nn.ModuleList([RawTransformerBlock(dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask_missing: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,T), mask_missing: (B,C) -> optional token mask
        z = self.patch(x)  # (B, N_t, D)
        # Simple: no per-token mask (tokens are time patches mixed across channels by conv)
        kpm = None
        for blk in self.blocks:
            z = blk(z, key_padding_mask=kpm)
        return self.norm(z)  # (B, N_t, D)


# ----------------------------
# Time–frequency branch
# ----------------------------

class LogSTFT(nn.Module):
    """Compute log-magnitude STFT per channel and stack."""

    def __init__(self, n_fft: int = 256, hop: int = 128, eps: float = 1e-6, center: bool = True):
        super().__init__()
        self.n_fft, self.hop, self.eps, self.center = n_fft, hop, eps, center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C, F, Tspec)
        B, C, T = x.shape
        device = x.device
        win = torch.hann_window(self.n_fft, device=device)
        specs = []
        for c in range(C):
            S = torch.stft(
                x[:, c],
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=win,
                return_complex=True,
                center=self.center,
            )
            mag = S.abs().clamp_min(self.eps)  # (B, F, Tspec)
            specs.append(torch.log1p(mag))
        spec = torch.stack(specs, dim=1)  # (B, C, F, Tspec)
        return spec


class ChannelReduce2D(nn.Module):
    """Reduce channel dimension (C -> out_ch) with 1x1 conv."""

    def __init__(self, in_ch: int, out_ch: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))  # (B, out_ch, F, Tspec)


class PatchEmbed2D(nn.Module):
    """2D patchify via Conv2d with stride=kernel=patch."""

    def __init__(self, in_ch: int = 4, dim: int = 256, patch: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,in_ch,F,T) -> (B,N_s,D)
        z = self.proj(x)  # (B, D, Fp, Tp)
        B, D, Fp, Tp = z.shape
        return z.flatten(2).transpose(1, 2)  # (B, N_s, D)


class ViTSmallEncoder(nn.Module):
    """Tiny ViT-like encoder reusing RawTransformerBlock."""

    def __init__(self, dim: int = 256, depth: int = 6, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([RawTransformerBlock(dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = tokens
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


class SpecBranch(nn.Module):
    """STFT -> 2D patch tokens -> transformer encoder."""

    def __init__(
            self,
            in_ch: int = 19,
            dim: int = 256,
            depth: int = 6,
            heads: int = 4,
            n_fft: int = 256,
            hop: int = 128,
            out_ch: int = 4,
            patch: Tuple[int, int] = (8, 8),
            drop: float = 0.1,
    ):
        super().__init__()
        self.tf = LogSTFT(n_fft, hop)
        self.reduce = ChannelReduce2D(in_ch, out_ch)
        self.patch = PatchEmbed2D(out_ch, dim, patch)
        self.encoder = ViTSmallEncoder(dim, depth, heads, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.tf(x)  # (B, C, F, Tspec)
        s = self.reduce(s)  # (B, out_ch, F, Tspec)
        tok = self.patch(s)  # (B, N_s, D)
        return self.encoder(tok)  # (B, N_s, D)


# ----------------------------
# Fusion + Heads
# ----------------------------

class CrossAttention(nn.Module):
    """Gated cross-attention Q<-K/V with residual MLP."""

    def __init__(self, dim: int, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=drop)
        self.gate_q = nn.Linear(dim, dim)
        self.gate_k = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.mlp = GatedFFN(dim, dropout=drop)
        self.drop = nn.Dropout(drop)

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gq = torch.sigmoid(self.gate_q(Q))
        gk = torch.sigmoid(self.gate_k(K))
        Qg = self.norm_q(Q * gq)
        Kg = self.norm_k(K * gk)
        out, _ = self.attn(Qg, Kg, Kg, key_padding_mask=key_padding_mask, need_weights=False)
        Z = Q + self.drop(out)
        Z = Z + self.mlp(Z)
        return Z


class AttentionPool(nn.Module):
    """Learned attention pooling over tokens (no CLS token)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B, D)
        B, N, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B,1,D)
        attn = torch.softmax((self.norm(q) @ self.norm(x).transpose(1, 2)) / (D ** 0.5), dim=-1)
        pooled = attn @ x  # (B,1,N)@(B,N,D)->(B,1,D)
        return pooled.squeeze(1)  # (B,D)


class FusionLayer(nn.Module):
    """Bi-directional cross-attention + pooling and concat."""

    def __init__(self, dim: int = 256, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.rs = CrossAttention(dim, heads, drop)  # raw <- spec
        self.sr = CrossAttention(dim, heads, drop)  # spec <- raw
        self.pool_r = AttentionPool(dim)
        self.pool_s = AttentionPool(dim)
        self.post = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2 * dim), nn.GELU())

    def forward(
            self,
            R: torch.Tensor,  # (B, Nt, D)
            S: torch.Tensor,  # (B, Ns, D)
            r_mask: Optional[torch.Tensor] = None,
            s_mask: Optional[torch.Tensor] = None,
    ):
        Rf = self.rs(R, S, key_padding_mask=s_mask)  # (B, Nt, D)
        Sf = self.sr(S, R, key_padding_mask=r_mask)  # (B, Ns, D)
        r_emb = self.pool_r(Rf)  # (B, D)
        s_emb = self.pool_s(Sf)  # (B, D)
        fused = self.post(torch.cat([r_emb, s_emb], dim=-1))  # (B, 2D)
        return fused, Rf, Sf


class SSLHeads(nn.Module):
    """Lightweight placeholders for masked modeling & InfoNCE (not used in forecast-only run)."""

    def __init__(self, dim: int = 256, proj: int = 128):
        super().__init__()
        self.raw_pred = nn.Linear(dim, dim)
        self.spec_pred = nn.Linear(dim, dim)
        self.proj_q = nn.Linear(dim, proj)
        self.proj_k = nn.Linear(dim, proj)

    def info_nce(self, r_emb: torch.Tensor, s_emb: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        q = F.normalize(self.proj_q(r_emb), dim=-1)
        k = F.normalize(self.proj_k(s_emb), dim=-1)
        sim = q @ k.t() / temp
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)


class ForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class HazardHead(nn.Module):
    """Discrete-time survival (hazard per bin)."""

    def __init__(self, in_dim: int, K: int = 12, hidden: int = 256, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, K)


# ----------------------------
# Public config + wrapper
# ----------------------------

@dataclass
class EFMConfig:
    in_ch: int = 19
    dim: int = 256
    raw_depth: int = 6
    spec_depth: int = 6
    heads: int = 4
    drop: float = 0.1
    patch_len: int = 64
    n_fft: int = 256
    hop: int = 128
    spec_ch: int = 4
    patch2d: Tuple[int, int] = (8, 8)
    proj_dim: int = 128
    cls_hidden: int = 256
    use_hazard: bool = False
    hz_bins: int = 12


class EFM(nn.Module):
    """End-to-end model that returns fused embeddings + task heads."""

    def __init__(self, cfg: EFMConfig):
        super().__init__()
        D = cfg.dim
        self.raw = RawBranch(cfg.in_ch, D, cfg.raw_depth, cfg.patch_len, cfg.heads, cfg.drop)
        self.spec = SpecBranch(cfg.in_ch, D, cfg.spec_depth, cfg.heads, cfg.n_fft, cfg.hop, cfg.spec_ch, cfg.patch2d,
                               cfg.drop)
        self.fusion = FusionLayer(D, cfg.heads, cfg.drop)
        self.ssl = SSLHeads(D, cfg.proj_dim)
        self.forecast = ForecastHead(in_dim=2 * D, hidden=cfg.cls_hidden)
        self.hazard = HazardHead(in_dim=2 * D, K=cfg.hz_bins, hidden=cfg.cls_hidden) if cfg.use_hazard else None

    def forward(
            self,
            x: torch.Tensor,  # (B, C, T)
            mask_missing: Optional[torch.Tensor] = None,
            train_ssl: bool = False,
            hz_labels: Optional[torch.Tensor] = None,
    ):
        R = self.raw(x, mask_missing)  # (B, Nt, D)
        S = self.spec(x)  # (B, Ns, D)
        fused, Rf, Sf = self.fusion(R, S)

        out = {
            "fused_emb": fused,  # (B, 2D)
            "forecast_logit": self.forecast(fused),  # (B,)
        }
        if self.hazard is not None:
            out["hazard_logits"] = self.hazard(fused)  # (B, K)
        if train_ssl:
            out["ssl"] = {"R_tokens": Rf, "S_tokens": Sf}
        return out
