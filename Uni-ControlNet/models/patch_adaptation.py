# /data/coding/Uni-ControlNet/models/patch_adaptation.py
# -*- coding: utf-8 -*-
"""
Patch Adaptation Module (PAM) for Uni-ControlNet (LocalAdapter FeatureExtractor).

あなたの要件（状況４〜６）に合わせた設計：
- 条件別Stem（edge/depth/seg）で FeatureExtractor 前半を独立化
- Tri-Attentionは「どれを選ぶか（logit計算）」にのみ使用（特徴融合は禁止）
- Hard選択(Argmax)を forward に使い、backward は softmax 勾配（STE）
- time-aware / text-aware を logit に入れる
- optional enhancer（キメラ後の特徴抽出を補強）を任意で挿入可能
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import timestep_embedding

logger = logging.getLogger(__name__)


def _ste_one_hot_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = 1,
) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) for hard categorical selection.

    Args:
        logits: (B, K, H, W) など。dim が K 次元
        temperature: softmax 温度 τ
        dim: softmax/argmax を取る次元（通常 K 次元）

    Returns:
        weights: (B, K, H, W)
        - forward: hard one-hot (argmax)
        - backward: softmax の勾配
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    w_soft = torch.softmax(logits / temperature, dim=dim)
    idx = torch.argmax(logits, dim=dim)

    # one_hot は最後に K が来るので permute が必要
    w_hard = F.one_hot(idx, num_classes=logits.shape[dim]).to(dtype=logits.dtype)

    if logits.dim() == 4 and dim == 1:
        # logits=(B,K,H,W), idx=(B,H,W), one_hot=(B,H,W,K) -> (B,K,H,W)
        w_hard = w_hard.permute(0, 3, 1, 2).contiguous()
    elif logits.dim() == 3 and dim == 1:
        # logits=(B,K,L), idx=(B,L), one_hot=(B,L,K) -> (B,K,L)
        w_hard = w_hard.permute(0, 2, 1).contiguous()
    else:
        raise NotImplementedError(
            f"_ste_one_hot_from_logits only supports logits dim=3/4 with dim=1. "
            f"Got logits.shape={tuple(logits.shape)}, dim={dim}"
        )

    # STE
    w = (w_hard - w_soft).detach() + w_soft
    return w


class ResidualConvBlock(nn.Module):
    """
    キメラ後の特徴抽出を補強するための、軽量な残差Convブロック。
    """

    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.groups = int(groups)

        self.norm1 = nn.GroupNorm(num_groups=self.groups, num_channels=self.channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=self.groups, num_channels=self.channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ChimeraEnhancer(nn.Module):
    """
    キメラ後の特徴抽出が弱くなる懸念に対応するための追加アーキテクチャ。

    - enable=False なら identity
    - enable=True なら ResidualConvBlock を n_blocks 個適用
    """

    def __init__(
        self,
        channels: int,
        enable: bool = False,
        n_blocks: int = 2,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        self.channels = int(channels)
        self.n_blocks = int(n_blocks)
        self.groups = int(groups)

        if not self.enable:
            self.net = nn.Identity()
        else:
            blocks = []
            for _ in range(self.n_blocks):
                blocks.append(ResidualConvBlock(channels=self.channels, groups=self.groups))
            self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerConditionStem(nn.Module):
    """
    FeatureExtractor前半を条件ごとに独立化したStem。

    元のFeatureExtractorに合わせた解像度遷移:
        input (B,3,512,512)
        -> 32ch @512
        -> 64ch @256 (stride2)
        -> 64ch @256
        -> 128ch@128 (stride2)
        -> 128ch@128
        -> 192ch@64 (stride2)  ★ここが “キメラ前”
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 192, kernel_size=3, padding=1, stride=2)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))  # -> 256
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))  # -> 128
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))  # -> 64
        return x  # (B,192,64,64)


class TriAttention(nn.Module):
    """
    “本物の tri-attention” （query × key × context の三者相互作用）

    あなたが提示したお手本実装の核：
        key_context   = key   * context
        value_context = value * context
    を multi-head に拡張したもの。

    重要：
    - これは “logit計算用” のみ（特徴融合して後段へ流す用途では使わない）
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads. embed_dim={embed_dim}, num_heads={num_heads}")
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.context = nn.Linear(self.embed_dim, self.embed_dim)

        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, H, L, Dh)
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, Dh) -> (B, L, D)
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)
        return x

    def forward(self, tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, K, D)   # K=3 条件token
            ctx:    (B, 1, D)   # time+text などの文脈token（1個で十分）

        Returns:
            out: (B, K, D)
        """
        q = self._split_heads(self.query(tokens))
        k = self._split_heads(self.key(tokens))
        v = self._split_heads(self.value(tokens))
        c = self._split_heads(self.context(ctx))  # (B,H,1,Dh)

        # tri-linear modulation（お手本実装に対応）
        # key_context:   (B,H,K,Dh) * (B,H,1,Dh) -> (B,H,K,Dh)
        # value_context: (B,H,K,Dh) * (B,H,1,Dh) -> (B,H,K,Dh)
        k_ctx = k * c
        v_ctx = v * c

        # attention
        scores = torch.matmul(q, k_ctx.transpose(-1, -2))  # (B,H,K,K)
        scores = scores / (self.head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v_ctx)  # (B,H,K,Dh)
        out = self._merge_heads(out)      # (B,K,D)
        out = self.out(out)               # (B,K,D)
        return out


@dataclass
class PAMConfig:
    # channel slices for local_conditions (B,21,H,W)
    # 公式順序: [canny, mlsd, hed, sketch, openpose, midas, seg] ×3ch
    edge_slice: Tuple[int, int] = (0, 3)
    depth_slice: Tuple[int, int] = (15, 18)
    seg_slice: Tuple[int, int] = (18, 21)

    # gating
    model_channels: int = 320   # timestep_embedding の次元（LocalAdapter model_channels と一致）
    context_dim: int = 768      # CLIP text embedding dim
    gating_dim: int = 192       # Stem出力(ch=192)に合わせる
    num_heads: int = 8
    attn_dropout: float = 0.0
    temperature: float = 1.0

    # logging
    log_every: int = 500

    # post-chimera enhancer
    enhancer_enable: bool = False
    enhancer_n_blocks: int = 2
    enhancer_groups: int = 32


class PatchPonderPAM(nn.Module):
    """
    PAM 本体:
    - 3条件 Stem
    - Tri-Attention gating (logit計算のみ)
    - STEでHard選択して chimera feature を返す
    - optional enhancer を chimera に適用
    """

    def __init__(self, cfg: PAMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 3条件 Stem
        self.stem_edge = PerConditionStem(in_channels=3)
        self.stem_depth = PerConditionStem(in_channels=3)
        self.stem_seg = PerConditionStem(in_channels=3)

        # time/text embedding -> gating_dim(=192)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.cfg.model_channels, self.cfg.gating_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.gating_dim, self.cfg.gating_dim),
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(self.cfg.context_dim, self.cfg.gating_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.gating_dim, self.cfg.gating_dim),
        )

        # token projection + tri-attn
        self.token_in = nn.Linear(self.cfg.gating_dim, self.cfg.gating_dim)
        self.ln1 = nn.LayerNorm(self.cfg.gating_dim)

        self.tri_attn = TriAttention(
            embed_dim=self.cfg.gating_dim,
            num_heads=self.cfg.num_heads,
            dropout=self.cfg.attn_dropout,
        )

        self.ln2 = nn.LayerNorm(self.cfg.gating_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.cfg.gating_dim, self.cfg.gating_dim * 4),
            nn.SiLU(),
            nn.Linear(self.cfg.gating_dim * 4, self.cfg.gating_dim),
        )
        self.gate_out = nn.Linear(self.cfg.gating_dim, 1)

        # optional enhancer
        self.enhancer = ChimeraEnhancer(
            channels=self.cfg.gating_dim,
            enable=self.cfg.enhancer_enable,
            n_blocks=self.cfg.enhancer_n_blocks,
            groups=self.cfg.enhancer_groups,
        )

        # init flag (0: not initialized, 1: initialized)
        self.register_buffer("initialized_from_base", torch.zeros((), dtype=torch.long))
        self.register_buffer("_iter", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def init_from_unicontrol_feature_extractor(self, feature_extractor: nn.Module) -> None:
        """
        既存の Uni-ControlNet FeatureExtractor (pre_extractor + extractors[0]) から
        Stem に重みをコピーして初期化する。
        """
        if self.initialized_from_base.item() == 1:
            return

        def _collect_convs(m: nn.Module) -> List[nn.Conv2d]:
            convs = []
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    convs.append(layer)
            return convs

        if not hasattr(feature_extractor, "pre_extractor"):
            logger.warning("[PatchPonderPAM] base feature_extractor has no pre_extractor. Skip init.")
            return
        base_pre = feature_extractor.pre_extractor
        base_pre_convs = _collect_convs(base_pre)
        if len(base_pre_convs) < 5:
            logger.warning(f"[PatchPonderPAM] base_pre_convs len={len(base_pre_convs)} < 5. Skip init.")
            return

        base_c1, base_c2, base_c3, base_c4, base_c5 = base_pre_convs[:5]

        if not hasattr(feature_extractor, "extractors") or len(feature_extractor.extractors) < 1:
            logger.warning("[PatchPonderPAM] base feature_extractor has no extractors[0]. Skip init.")
            return
        base_ext0 = feature_extractor.extractors[0]
        base_ext0_convs = _collect_convs(base_ext0)
        if len(base_ext0_convs) < 1:
            logger.warning("[PatchPonderPAM] base_ext0 has no conv. Skip init.")
            return
        base_c6 = base_ext0_convs[0]  # 128->192

        def _copy_to_stem(stem: PerConditionStem, in_slice: Tuple[int, int]) -> None:
            s0, s1 = in_slice
            if base_c1.weight.shape[1] >= s1 and (s1 - s0) == 3:
                stem.conv1.weight.copy_(base_c1.weight[:, s0:s1, :, :])
                if base_c1.bias is not None and stem.conv1.bias is not None:
                    stem.conv1.bias.copy_(base_c1.bias)
            else:
                logger.warning(
                    f"[PatchPonderPAM] conv1 slice mismatch. base={tuple(base_c1.weight.shape)} slice={in_slice}. "
                    "Skip conv1 init."
                )

            # conv2..5
            stem.conv2.weight.copy_(base_c2.weight); stem.conv2.bias.copy_(base_c2.bias)
            stem.conv3.weight.copy_(base_c3.weight); stem.conv3.bias.copy_(base_c3.bias)
            stem.conv4.weight.copy_(base_c4.weight); stem.conv4.bias.copy_(base_c4.bias)
            stem.conv5.weight.copy_(base_c5.weight); stem.conv5.bias.copy_(base_c5.bias)
            # conv6
            stem.conv6.weight.copy_(base_c6.weight); stem.conv6.bias.copy_(base_c6.bias)

        _copy_to_stem(self.stem_edge, self.cfg.edge_slice)
        _copy_to_stem(self.stem_depth, self.cfg.depth_slice)
        _copy_to_stem(self.stem_seg, self.cfg.seg_slice)

        self.initialized_from_base.fill_(1)
        logger.info("[PatchPonderPAM] Initialized stems from base FeatureExtractor weights.")

    def _compute_logits(
        self,
        f_edge: torch.Tensor,
        f_depth: torch.Tensor,
        f_seg: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tri-Attention gating で logits を作る。

        Returns:
            logits: (B, K=3, H, W)
        """
        B, C, H, W = f_edge.shape
        P = H * W
        K = 3

        # (B,C,H,W) -> (B,P,C)
        e = f_edge.permute(0, 2, 3, 1).reshape(B, P, C)
        d = f_depth.permute(0, 2, 3, 1).reshape(B, P, C)
        s = f_seg.permute(0, 2, 3, 1).reshape(B, P, C)

        # (B,P,K,C) -> (B*P,K,C)
        tokens = torch.stack([e, d, s], dim=2).reshape(B * P, K, C)

        # time embedding (B,model_channels)->(B,gating_dim)
        t_emb = timestep_embedding(timesteps, self.cfg.model_channels, repeat_only=False).to(tokens.dtype)
        t_vec = self.time_mlp(t_emb)  # (B,192)

        # text embedding (B,seq,768)->mean->(B,768)->(B,192)
        if context.dim() != 3:
            raise ValueError(f"context must be (B,seq,dim). Got {tuple(context.shape)}")
        txt_vec_in = context.mean(dim=1)
        txt_vec = self.txt_mlp(txt_vec_in.to(tokens.dtype))  # (B,192)

        # ctx token: (B,192) -> (B*P,1,192)
        ctx = (t_vec + txt_vec).repeat_interleave(P, dim=0).unsqueeze(1)

        # tri-attn (logit calc only)
        x = self.token_in(tokens)
        x = self.ln1(x)

        attn_out = self.tri_attn(x, ctx)      # (B*P,K,192)
        x = x + attn_out
        x = self.ln2(x)
        x = x + self.ff(x)

        logits = self.gate_out(x).squeeze(-1)  # (B*P,K)

        # (B,K,H,W)
        logits = logits.reshape(B, P, K).permute(0, 2, 1).reshape(B, K, H, W).contiguous()
        return logits

    def forward(
        self,
        local_conditions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        base_feature_extractor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            local_conditions: (B,21,512,512) float in [0,1]
            timesteps: (B,) long
            context: (B,seq,768) text context
            base_feature_extractor:
                FeatureExtractor本体を渡すと、初回だけそこからstemを初期化する。

        Returns:
            chimera: (B,192,64,64)
        """
        self._iter += 1

        if (self.initialized_from_base.item() == 0) and (base_feature_extractor is not None):
            self.init_from_unicontrol_feature_extractor(base_feature_extractor)

        if local_conditions.dim() != 4:
            raise ValueError(f"local_conditions must be 4D (B,C,H,W). Got {tuple(local_conditions.shape)}")
        if local_conditions.shape[1] < 21:
            raise ValueError(f"local_conditions channel must be >=21. Got {local_conditions.shape[1]}")

        e0, e1 = self.cfg.edge_slice
        d0, d1 = self.cfg.depth_slice
        s0, s1 = self.cfg.seg_slice

        edge = local_conditions[:, e0:e1, :, :]
        depth = local_conditions[:, d0:d1, :, :]
        seg = local_conditions[:, s0:s1, :, :]

        # stems -> (B,192,64,64)
        f_edge = self.stem_edge(edge)
        f_depth = self.stem_depth(depth)
        f_seg = self.stem_seg(seg)

        # logits -> weights(STE) -> chimera
        logits = self._compute_logits(
            f_edge=f_edge,
            f_depth=f_depth,
            f_seg=f_seg,
            timesteps=timesteps,
            context=context,
        )
        weights = _ste_one_hot_from_logits(logits, temperature=self.cfg.temperature, dim=1)  # (B,3,64,64)

        chimera = (
            weights[:, 0:1, :, :] * f_edge
            + weights[:, 1:2, :, :] * f_depth
            + weights[:, 2:3, :, :] * f_seg
        )

        chimera = self.enhancer(chimera)

        # logging
        if (self.cfg.log_every > 0) and (int(self._iter.item()) % int(self.cfg.log_every) == 0):
            w_mean = weights.mean(dim=(0, 2, 3)).detach().cpu().tolist()
            w_std = weights.std(dim=(0, 2, 3)).detach().cpu().tolist()
            logger.info(
                "[PatchPonderPAM] iter=%d weights mean(edge,depth,seg)=%s std=%s chimera_shape=%s",
                int(self._iter.item()),
                str(w_mean),
                str(w_std),
                str(tuple(chimera.shape)),
            )

        return chimera
