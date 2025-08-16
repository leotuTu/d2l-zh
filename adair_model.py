# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# ============================ LayerNorm ============================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ======================= Transformer Primitives =======================
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw          = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dw(x).chunk(2, 1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads  = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv       = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw    = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.out       = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, 1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = attn @ v
        out  = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================ Resizing & Embed ============================
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x): return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    def forward(self, x): return self.proj(x)


# =================== Cross-Attention (channel-wise) primitive ===================
class _QKVPath(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dwc3  = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
    def forward(self, x):
        return self.dwc3(self.conv1(x))

def channel_cross_attention(Qx, Ky, Vy, temperature=None):
    B, C, H, W = Qx.shape
    HW = H * W
    q = Qx.view(B, C, HW)
    k = Ky.view(B, C, HW)
    v = Vy.view(B, C, HW)
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
    attn = torch.matmul(q, k.transpose(1, 2)) / (HW ** 0.5 + 1e-6)  # [B,C,C]
    if temperature is not None:
        attn = attn * temperature
    attn = attn.softmax(dim=-1)
    out = torch.matmul(attn, v).view(B, C, H, W)
    return out


# ===================== 分离实现：Decoupler 与 Modulator =====================
class GaussianDecoupler(nn.Module):
    def __init__(self, dim, ksize=31, base_sigmas=(0.8, 1.6, 3.2), bias=False):
        super().__init__()
        assert ksize % 2 == 1
        self.dim = dim
        self.ksize = ksize
        M = len(base_sigmas)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Sequential(
            nn.Conv2d(dim, M, 1, bias=bias),
            nn.BatchNorm2d(M),
        )
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer('gauss_bases', self._build_gaussian_bases(ksize, base_sigmas))

    @torch.no_grad()
    def _build_gaussian_bases(self, k, sigmas):
        xs = torch.arange(k) - (k // 2)
        yy, xx = torch.meshgrid(xs, xs, indexing='ij')
        yy, xx = yy.float(), xx.float()
        bases = []
        for s in sigmas:
            g = torch.exp(-(xx**2 + yy**2) / (2 * (s**2)))
            g = g / g.sum()
            bases.append(g)
        return torch.stack(bases, 0)  # [M,k,k]

    def _make_kernel(self, w):  # w: [B,M,1,1]
        B, M, _, _ = w.shape
        K = torch.tensordot(w.view(B, M), self.gauss_bases, dims=([1], [0]))  # [B,k,k]
        K = K / (K.sum(dim=[1, 2], keepdim=True) + 1e-12)
        return K.view(B, 1, self.ksize, self.ksize)

    def _depthwise_per_sample(self, x, K):
        pad = self.ksize // 2
        B, C, H, W = x.shape
        outs = []
        for b in range(B):
            w = K[b:b+1].repeat(C, 1, 1, 1)
            y = F.conv2d(x[b:b+1], w, padding=pad, groups=C)
            outs.append(y)
        return torch.cat(outs, 0)

    def forward(self, Fs):
        w = self.softmax(self.selector(self.gap(Fs)))     # [B,M,1,1]
        K = self._make_kernel(w)                          # [B,1,k,k]
        Fl = self._depthwise_per_sample(Fs, K)            # 平滑/低频
        Fh = Fs - Fl                                      # 高频残差
        return Fl, Fh


# --------------------- Cross-Attention（用于 DCM） ---------------------
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim: int, num_head: int = 4, bias: bool = False):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        # Q path
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # KV path
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # projection after attention
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'
        b, c, h, w = x.shape
        assert c % self.num_head == 0, "Channel dim must be divisible by num_head."
        c1 = c // self.num_head

        q = self.q_dwconv(self.q(x))           # [B,C,H,W]
        kv = self.kv_dwconv(self.kv(y))        # [B,2C,H,W]
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) H W -> b head c (H W)', head=self.num_head, c=c1)
        k = rearrange(k, 'b (head c) H W -> b head c (H W)', head=self.num_head, c=c1)
        v = rearrange(v, 'b (head c) H W -> b head c (H W)', head=self.num_head, c=c1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = (h * w) ** -0.5

        attn = (q @ k.transpose(-2, -1)) * self.temperature * scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (H W) -> b (head c) H W', head=self.num_head, H=h, W=w)
        out = self.project_out(out)
        return out


# --------------------- LEU / HEU 预调制 ---------------------
class LEU(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, bias: bool = True):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=(2, 3), keepdim=True)
        gmp = x.amax(dim=(2, 3), keepdim=True)
        gsp = x.float().std(dim=(2, 3), keepdim=True, unbiased=False).to(x.dtype)
        y = self.mlp(gap) + self.mlp(gmp) + self.mlp(gsp)
        w = self.gate(y)
        return x * w

class HEU(nn.Module):
    def __init__(self, bias: bool = True):
        super().__init__()
        self.conv_1x7 = nn.Conv2d(2, 1, kernel_size=(1, 7), padding=(0, 3), bias=bias)
        self.conv_7x1 = nn.Conv2d(2, 1, kernel_size=(7, 1), padding=(3, 0), bias=bias)
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=1, keepdim=True)
        gmp = x.amax(dim=1, keepdim=True)
        s = torch.cat([gap, gmp], dim=1)
        s1 = self.conv_1x7(s)
        s2 = self.conv_7x1(s)
        w = self.gate(self.fuse(torch.cat([s1, s2], dim=1)))
        # return x * w
        # HEU 的最后一行改：
        return x + 0.2 * (x * w)


class BidirectionalModulator(nn.Module):
    def __init__(self, dim, bias=False, num_heads: int = 4, leu_reduction: int = 4):
        super().__init__()
        # 预调制
        self.leu = LEU(dim, reduction=leu_reduction, bias=True)
        self.heu = HEU(bias=True)

        # 双向交叉注意力
        self.ca_hl = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)  # Q=Fh, KV=Fl
        self.ca_lh = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)  # Q=Fl, KV=Fh

        # 融合：concat 后 1×1
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 残差恒等频域门（幅度门）
        self.fmg = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Tanh()   # 输出在（-1，1）
        )
        # 门强度（可学习缩放，初始很小，近似恒等）
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可在 0.05~0.2

        # 把CA交互结果当成残差增益，初始≃0.5×，更稳。
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, Fl, Fh):
        # 预调制
        Fl_t = self.leu(Fl)
        Fh_t = self.heu(Fh)

        # 双向交叉注意力
        # F_hl = self.ca_hl(Fh_t, Fl_t)                   # Q=Fh_t, K/V=Fl_t
        # F_lh = self.ca_lh(Fl_t, Fh_t)                   # Q=Fl_t, K/V=Fh_t
        # F_fuse = self.fuse(torch.cat([F_hl, F_lh], dim=1))

        F_hl = Fh_t + torch.sigmoid(self.beta) * (self.ca_hl(Fh_t, Fl_t))
        F_lh = Fl_t + torch.sigmoid(self.beta) * (self.ca_lh(Fl_t, Fh_t))
        F_fuse = self.fuse(torch.cat([F_hl, F_lh], dim=1))

        # --- 在 log|F| 上做小幅残差调制，初始≈恒等 ---
        F_freq = torch.fft.fft2(F_fuse, norm='ortho')   # complex
        mag = torch.abs(F_freq).clamp_min(1e-6)
        # 归一化与对数稳定化，抑制亮度/对比度差异的漂移
        mag_log = torch.log1p(mag)
        mag_log = (mag_log - mag_log.mean(dim=(2, 3), keepdim=True)) / (mag_log.std(dim=(2, 3), keepdim=True) + 1e-5)

        G = self.fmg(mag_log)  # (-1,1)
        M_freq = 1.0 + torch.tanh(self.alpha) * G  # 初始≈1，微调
        F_mod = F_freq * M_freq.to(F_freq.dtype)

        I = torch.fft.ifft2(F_mod, norm='ortho').real

        return I + F_fuse  # 残差


# ===================== 封装类：Decouple-Then-Modulate =====================
class DecoupleThenModulate(nn.Module):
    def __init__(self, dim, ksize=31, base_sigmas=(0.8, 1.6, 3.2), bias=False):
        super().__init__()
        self.decoupler = GaussianDecoupler(dim, ksize=ksize, base_sigmas=base_sigmas, bias=bias)
        self.modulator = BidirectionalModulator(dim, bias=bias)

    def forward(self, Fs):
        Fl, Fh = self.decoupler(Fs)
        return self.modulator(Fl, Fh)


# ================================ AdaIR（接单块） ================================
class AdaIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 decoder=True):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        # ---------- 这里用"封装类"保持原位调用 ----------
        if self.decoder:
            self.dcm1 = DecoupleThenModulate(dim * 2**3, bias=bias)
            self.dcm2 = DecoupleThenModulate(dim * 2**2, bias=bias)
            self.dcm3 = DecoupleThenModulate(dim * 2**1, bias=bias)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2**1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 2**2))

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**3), num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])
        ])

        # Decoder
        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), 1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), 1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        # Encoder
        x1 = self.patch_embed(inp_img)
        e1 = self.encoder_level1(x1)

        x2 = self.down1_2(e1)
        e2 = self.encoder_level2(x2)

        x3 = self.down2_3(e2)
        e3 = self.encoder_level3(x3)

        x4 = self.down3_4(e3)
        lat = self.latent(x4)

        # DCM@latent
        if self.decoder:
            lat = self.dcm1(lat)

        # Decoder level3
        d3 = self.up4_3(lat)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.reduce_chan_level3(d3)
        d3 = self.decoder_level3(d3)

        # DCM@dec3
        if self.decoder:
            d3 = self.dcm2(d3)

        # Decoder level2
        d2 = self.up3_2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.reduce_chan_level2(d2)
        d2 = self.decoder_level2(d2)

        # DCM@dec2
        if self.decoder:
            d2 = self.dcm3(d2)

        # Decoder level1
        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.decoder_level1(d1)

        out = self.refinement(d1)
        out = self.output(out) + inp_img
        return out


# =============================== quick self-test ===============================
if __name__ == "__main__":
    with torch.no_grad():
        model = AdaIR(dim=32).eval()
        x = torch.rand(1, 3, 128, 128)
        y = model(x)
        print("output:", y.shape)  # [1,3,128,128]