"""
MambaOut models for image classification.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""

from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode
from .operator import XmmLayer

Lif = partial(
    LIFNode,
    tau=2.0,
    backend="cupy",
    step_mode="m",
    surrogate_function=surrogate.ATan(),
)


T = 1


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 100,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "mambaout_femto": _cfg(
        url="https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth"
    ),
    "mambaout_pico": _cfg(url=""),
    "mambaout_kobe": _cfg(
        url="https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth"
    ),
    "mambaout_tiny": _cfg(
        url="https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth"
    ),
    "mambaout_small": _cfg(
        url="https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth"
    ),
    "mambaout_base": _cfg(
        url="https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth"
    ),
}


class StemLayer(nn.Module):
    r"""Code modified from InternImage:
    https://github.com/OpenGVLab/InternImage
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=96,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_Lif=False,
    ):
        super().__init__()
        self.use_Lif = use_Lif
        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = norm_layer(out_channels // 2)
        self.lif1 = Lif() if use_Lif else act_layer()
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.norm2 = norm_layer(out_channels)
        self.lif2 = Lif() if use_Lif else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.lif1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.lif2(x)
        return x


class DownsampleLayer(nn.Module):
    r"""Code modified from InternImage:
    https://github.com/OpenGVLab/InternImage
    """

    def __init__(
        self,
        in_channels=96,
        out_channels=198,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_Lif=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.norm = norm_layer(out_channels)
        self.lif = Lif() if use_Lif else nn.Identity()

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.lif(x)
        return x


class MlpHead(nn.Module):
    """MLP classification head"""

    def __init__(
        self,
        dim,
        num_classes=100,
        mlp_ratio=4,
        head_dropout=0.0,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        # normalization layers is inside XmmLayer.
        self.layers = nn.Sequential(
            XmmLayer(dim, hidden_features),
            nn.Dropout(head_dropout),
            XmmLayer(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
        self,
        dim,
        expansion_ratio=8 / 3,
        kernel_size=7,
        conv_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_Lif=False,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.lifc = Lif() if use_Lif else nn.Identity()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Linear(hidden, dim)
        self.lifg = Lif() if use_Lif else act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = self.lifc(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.lifg(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut


r"""
downsampling (stem) for the first stage is two layer of conv with k3, s2 and p1
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer] * 3


class MambaOut(nn.Module):
    r"""MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [3, 3, 9, 3].
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 576].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
        head_dropout (float): dropout for MLP classifier. Default: 0.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=100,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 576],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        conv_ratio=1.0,
        kernel_size=7,
        drop_path_rate=0.0,
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        head_dropout=0.0,
        use_Lif=True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_Lif = use_Lif

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [
                downsample_layers[i](down_dims[i], down_dims[i + 1])
                for i in range(num_stage)
            ]
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    GatedCNNBlock(
                        dim=dims[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        kernel_size=kernel_size,
                        conv_ratio=conv_ratio,
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2]))  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


###############################################################################
@register_model
def mambaout_pico(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 3, 9, 3], dims=[24, 48, 96, 128], use_lif=True, **kwargs
    )
    model.default_cfg = default_cfgs["mambaout_pico"]
    return model