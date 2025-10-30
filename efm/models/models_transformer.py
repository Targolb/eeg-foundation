# efm/models_transformer.py
import torch, torch.nn as nn, torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None


# ---------- RAW: Temporal Transformer ----------
class EFMRawTransformer(nn.Module):
    """
    Patchify time axis, keep channels as embedding features.
    x: (B, C, T)
    """

    def __init__(self, in_ch=19, d_model=512, n_heads=8, depth=6, patch_len=16, patch_stride=16, mlp_ratio=4.0,
                 dropout=0.1, out_dim=512):
        super().__init__()
        self.in_ch = in_ch
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=patch_stride, padding=0)  # (B,d_model,L)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=int(d_model * mlp_ratio),
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = None  # lazy
        self.head = nn.Linear(d_model, out_dim)

    def _build_pos(self, L, device):
        if self.pos is None or self.pos.size(1) != (L + 1):
            self.pos = nn.Parameter(torch.zeros(1, L + 1, self.cls.size(-1), device=device), requires_grad=True)

    def forward(self, x):  # (B,C,T)
        B, C, T = x.shape
        z = self.proj(x)  # (B, d_model, L)
        z = z.transpose(1, 2)  # (B, L, d_model)
        L = z.size(1)
        self._build_pos(L, x.device)
        cls = self.cls.expand(B, -1, -1)  # (B,1,d_model)
        z = torch.cat([cls, z], dim=1) + self.pos[:, :L + 1, :]
        z = self.encoder(z)  # (B, L+1, d_model)
        cls_out = z[:, 0]  # (B, d_model)
        return self.head(cls_out)  # (B, out_dim)

    # token features for masked patch prediction or CPC
    def forward_tokens(self, x):
        B, C, T = x.shape
        z = self.proj(x).transpose(1, 2)  # (B,L,d_model)
        L = z.size(1)
        self._build_pos(L, x.device)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos[:, :L + 1, :]
        z = self.encoder(z)  # (B,L+1,d_model)
        return z  # tokens (incl. CLS)


# ---------- SPEC: MaxViT over spectrogram ----------
class EFMSpecMaxViT(nn.Module):
    """
    Map EEG spec (B, C, F, T) -> 3x224x224 -> MaxViT backbone -> 512-D
    """

    def __init__(self, in_ch=19, out_dim=512, model_name='maxvit_tiny_tf_224'):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for MaxViT. pip install timm==0.9.12")
        self.rgb_map = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)  # global pooling -> feat
        feat_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 512
        self.head = nn.Linear(feat_dim, out_dim)

    def forward(self, spec):  # (B,C,F,T)
        x = self.rgb_map(spec)  # (B,3,F,T)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        f = self.backbone(x)  # (B, feat)
        return self.head(f)  # (B, out_dim)
