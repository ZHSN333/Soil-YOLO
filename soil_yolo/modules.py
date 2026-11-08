"""Soil-YOLO 轻量注意力 / 纹理增强 / 特征融合模块。

设计目标:
  - 强化颗粒、孔隙、粗糙度等纹理响应，避免仅依赖颜色
  - 保持输入输出 shape 完全一致: B,C,H,W -> B,C,H,W
  - 保持通道数不变，便于插入 YOLO11 Backbone / Neck
  - 参数量小，适配 YOLO11n / YOLO11s
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention.

    轻量通道注意力模块。
    输入输出形状:
        x: B,C,H,W
        y: B,C,H,W

    YAML:
        [-1, 1, ECA, [3]]
    """

    def __init__(self, k_size: int = 3):
        super().__init__()

        
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(k_size, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        
        y = self.avg_pool(x).view(b, 1, c)

        
        y = self.conv(y)

        
        y = y.view(b, c, 1, 1)

        return x * self.sigmoid(y)


class _MultiScaleDW(nn.Module):
    """多尺度 depthwise 卷积分支。

    branches:
      - "3"    : 仅 3x3
      - "3+5"  : 3x3 + 5x5
      - "full" : 3x3 + 5x5 + 3x3(dilation=3)

    输入:
        B,C,H,W
    输出:
        B,(nC),H,W  （n = 分支数）
    """

    def __init__(self, c: int, branches: str = "full"):
        super().__init__()
        mode = str(branches).lower().replace(" ", "")
        specs: list[tuple[int, int, int]] = []  
        if mode in {"3", "3x3"}:
            specs = [(3, 1, 1)]
        elif mode in {"3+5", "3x3+5x5"}:
            specs = [(3, 1, 1), (5, 2, 1)]
        else:  
            specs = [(3, 1, 1), (5, 2, 1), (3, 3, 3)]

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    c,
                    c,
                    kernel_size=k,
                    stride=1,
                    padding=p,
                    dilation=d,
                    groups=c,
                    bias=False,
                )
                for k, p, d in specs
            ]
        )
        self.num_branches = len(self.branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class _ReplicateBlur(nn.Module):
    """保持尺寸不变的局部均值滤波。

    相比 AvgPool2d(padding=1)，使用复制填充可以减轻边缘处的零填充伪影。
    输入输出:
        B,C,H,W -> B,C,H,W
    """

    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=0,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.pad(x))


def _zero_init_last_bn(module: nn.Module) -> None:
    """将模块中最后一个 BatchNorm2d 零初始化。

    作用:
      - 让残差分支初始输出接近 0
      - 插入已有 YOLO 网络时更稳定
      - 模块初始更接近恒等映射
    """

    last_bn = None
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            last_bn = m

    if last_bn is not None:
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)


class TextureEnhance(nn.Module):
    """轻量多尺度纹理增强模块。

    主要针对泥土图像中的:
      - 颗粒尺度差异
      - 孔隙纹理
      - 表面粗糙度
      - 结块结构

    输入输出形状:
        B,C,H,W -> B,C,H,W

    默认 YAML:
        [-1, 1, TextureEnhance, []]

    可选 YAML:
        [-1, 1, TextureEnhance, [None, 3, 0.1, "full", 1]]
        其中:
          c: Optional[int]，一般不用填
          k_size: ECA kernel size
          init_scale: 残差分支初始缩放系数
          branches: "3" | "3+5" | "full"
          use_hf: 1/0 是否启用高频残差
    """

    def __init__(
        self,
        c: Optional[int] = None,
        k_size: int = 3,
        init_scale: float = 0.1,
        branches: str = "full",
        use_hf: int = 1,
    ):
        super().__init__()

        self.c = c
        self.k_size = int(k_size)
        self.init_scale = float(init_scale)
        self.branches = str(branches)
        self.use_hf = bool(int(use_hf))

        self.ms_dw: Optional[nn.Module] = None
        self.fuse: Optional[nn.Module] = None
        self.eca: Optional[nn.Module] = None
        self.blur: Optional[nn.Module] = None

        self.tex_scale: Optional[nn.Parameter] = None
        self.hf_scale: Optional[nn.Parameter] = None

        if c is not None:
            self._build(c, device=torch.device("cpu"), dtype=torch.float32)

    def _build(self, c: int, device: torch.device, dtype: torch.dtype) -> None:
        self.c = c

        self.ms_dw = _MultiScaleDW(c, branches=self.branches)
        n_br = self.ms_dw.num_branches

        self.fuse = nn.Sequential(
            
            nn.Conv2d(c * n_br, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
        )

        self.eca = ECA(k_size=self.k_size)
        self.blur = _ReplicateBlur() if self.use_hf else None

        
        self.tex_scale = nn.Parameter(torch.tensor(self.init_scale, device=device, dtype=dtype))
        self.hf_scale = nn.Parameter(
            torch.tensor(self.init_scale * 0.25 if self.use_hf else 0.0, device=device, dtype=dtype)
        )

        
        _zero_init_last_bn(self.fuse)

        self.to(device=device, dtype=dtype)

    def _check_or_build(self, x: torch.Tensor) -> None:
        c = x.shape[1]

        if self.ms_dw is None:
            self._build(c, x.device, x.dtype)
            return

        if self.c != c:
            raise RuntimeError(
                f"TextureEnhance channel mismatch: "
                f"built with C={self.c}, but got C={c}. "
                f"请确保同一个 TextureEnhance 实例只用于固定通道数的特征层。"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_or_build(x)

        identity = x

        
        tex = self.ms_dw(x)
        tex = self.fuse(tex)
        tex = self.eca(tex)
        out = identity + self.tex_scale * tex

        
        use_hf = bool(getattr(self, "use_hf", True))
        blur = getattr(self, "blur", None)
        if use_hf and blur is not None:
            hf = x - blur(x)
            out = out + self.hf_scale * hf
        return out


class FusionRefine(nn.Module):
    """Neck 特征融合后的轻量 refinement 模块。

    适合放在 YOLO Neck 中 Concat / C2f / C3k2 后面，用于优化跨尺度融合后的特征。

    输入输出形状:
        B,C,H,W -> B,C,H,W

    YAML:
        [-1, 1, FusionRefine, []]
    """

    def __init__(
        self,
        c: Optional[int] = None,
        k_size: int = 3,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.c = c
        self.k_size = k_size
        self.init_scale = init_scale

        self.net: Optional[nn.Module] = None
        self.eca: Optional[nn.Module] = None
        self.gamma: Optional[nn.Parameter] = None

        if c is not None:
            self._build(c, device=torch.device("cpu"), dtype=torch.float32)

    def _build(self, c: int, device: torch.device, dtype: torch.dtype) -> None:
        self.c = c

        self.net = nn.Sequential(
            
            nn.Conv2d(
                c,
                c,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=c,
                bias=False,
            ),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(
                c,
                c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c),
        )

        self.eca = ECA(k_size=self.k_size)

        
        self.gamma = nn.Parameter(torch.tensor(self.init_scale, device=device, dtype=dtype))

        
        _zero_init_last_bn(self.net)

        self.to(device=device, dtype=dtype)

    def _check_or_build(self, x: torch.Tensor) -> None:
        c = x.shape[1]

        if self.net is None:
            self._build(c, x.device, x.dtype)
            return

        if self.c != c:
            raise RuntimeError(
                f"FusionRefine channel mismatch: "
                f"built with C={self.c}, but got C={c}. "
                f"请确保同一个 FusionRefine 实例只用于固定通道数的特征层。"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_or_build(x)

        identity = x
        y = self.net(x)
        y = self.eca(y)
        out = identity + self.gamma * y
        return out
