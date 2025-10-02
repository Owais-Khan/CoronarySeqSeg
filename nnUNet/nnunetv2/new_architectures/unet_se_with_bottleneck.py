import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
    get_matching_convtransp,
    convert_conv_op_to_dim
)

# ---------------------------------------------------------------------
# Basic conv block: Conv -> (Dropout) -> (Norm) -> (Nonlin)
# ---------------------------------------------------------------------
class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None: norm_op_kwargs = {}
        if nonlin_kwargs is None: nonlin_kwargs = {}

        ops = []
        self.conv = conv_op(
            input_channels, output_channels, kernel_size, stride,
            padding=[(k - 1) // 2 for k in kernel_size], dilation=1, bias=conv_bias
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x): return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride)
        # same padding → spatial size // stride
        out_sz = [i // j for i, j in zip(input_size, self.stride)]
        return np.prod([self.output_channels, *out_sz], dtype=np.int64)


# ---------------------------------------------------------------------
# SE (channel) + lightweight spatial gate (CBAM-style 1×k×k)
# ---------------------------------------------------------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, conv_op: Type[_ConvNd] = nn.Conv3d):
        super(SEBlock3D, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = max(1, channels // reduction_ratio)

        dim = convert_conv_op_to_dim(conv_op)
        if dim == 3:
            self.squeeze = nn.AdaptiveAvgPool3d(1)
        elif dim == 2:
            self.squeeze = nn.AdaptiveAvgPool2d(1)
        elif dim == 1:
            self.squeeze = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unsupported conv_op dimension for SEBlock: {dim}")

        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        self.dim = dim

    def forward(self, x: torch.Tensor, g: torch.Tensor = None):
        b, c, *spatial_dims = x.shape
        if c != self.channels:
            print(
                f"Warning (SEBlock3D): Input tensor x has {c} channels, but SEBlock was initialized with {self.channels} channels. This will likely cause an error in nn.Linear.")
            assert c == self.channels, "Channel mismatch in SEBlock excitation."

        feature_for_squeeze = x
        if g is not None:
            if x.shape[2:] != g.shape[2:]:
                interp_mode = 'trilinear' if self.dim == 3 else ('bilinear' if self.dim == 2 else 'linear')
                g_aligned = F.interpolate(g, size=x.shape[2:], mode=interp_mode, align_corners=False)
            else:
                g_aligned = g

            if x.shape[1] != g_aligned.shape[1]:
                print(
                    f"Warning (SEBlock3D): Channels of x ({x.shape[1]}) and g ({g_aligned.shape[1]}) for addition differ. Ensure channel counts match or implement projection for g.")
            else:  # Channels match
                feature_for_squeeze = x + g_aligned

        y = self.squeeze(feature_for_squeeze).view(b, c)
        y_excited = self.excitation(y)

        scale_factors_shape = [b, c] + [1] * self.dim
        scale_factors = y_excited.view(*scale_factors_shape)

        return x * scale_factors.expand_as(x)



class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        blocks = [
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            )
        ]
        for i in range(1, num_convs):
            blocks.append(
                ConvDropoutNormReLU(
                    conv_op, output_channels[i-1], output_channels[i], kernel_size, 1,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, nonlin_first
                )
            )
        self.convs = nn.Sequential(*blocks)
        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x): return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride)
        out = self.convs[0].compute_conv_feature_map_size(input_size)
        sz = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            out += b.compute_conv_feature_map_size(sz)
        return out


class ASPP(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 in_channels: int,
                 out_channels: int = None,
                 dilation_rates: Tuple[int, ...] = (6, 12, 18),
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        dim = convert_conv_op_to_dim(conv_op)
        out_channels = in_channels if out_channels is None else out_channels

        # 1x1 branch
        self.b0 = conv_op(in_channels, out_channels, kernel_size=1, bias=conv_bias)

        # Atrous branches
        self.b_dil = nn.ModuleList([
            conv_op(in_channels, out_channels, kernel_size=3,
                    padding=rate, dilation=rate, bias=conv_bias)
            for rate in dilation_rates
        ])

        # Image pooling branch
        if dim == 3:
            pool = nn.AdaptiveAvgPool3d(1)
            up_mode = 'trilinear'
        elif dim == 2:
            pool = nn.AdaptiveAvgPool2d(1)
            up_mode = 'bilinear'
        else:
            pool = nn.AdaptiveAvgPool1d(1)
            up_mode = 'linear'
        self.image_pool = nn.Sequential(
            pool,
            conv_op(in_channels, out_channels, kernel_size=1, bias=conv_bias)
        )
        self._up_mode = up_mode
        self._dim = dim

        # Fuse
        total = out_channels * (2 + len(dilation_rates))  # 1x1 + pooled + dilated branches
        self.proj = conv_op(total, out_channels, kernel_size=1, bias=conv_bias)

        # Optional norm/nonlin after projection
        post = []
        if norm_op is not None:
            post.append(norm_op(out_channels, **(norm_op_kwargs or {})))
        if nonlin is not None:
            post.append(nonlin(**(nonlin_kwargs or {})))
        self.post = nn.Sequential(*post)

    def forward(self, x):
        size = x.shape[2:]
        outs = [self.b0(x)]
        outs += [b(x) for b in self.b_dil]

        # pooled branch -> upsample to size
        p = self.image_pool(x)
        if self._dim == 1:
            p = F.interpolate(p, size=size[0], mode=self._up_mode, align_corners=False)
        else:
            p = F.interpolate(p, size=size, mode=self._up_mode, align_corners=False)
        outs.append(p)

        y = torch.cat(outs, dim=1)
        y = self.proj(y)
        y = self.post(y)
        return y

    def compute_conv_feature_map_size(self, input_size):
        def vol(sz): return int(np.prod(sz))
        return np.int64(0)

class PlainConvEncoder_se(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = True,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 se_reduction_ratio: int = 16):
        super().__init__()
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.nonlin_first = nonlin_first
        self.strides = strides  # original arg

        ks_per = [kernel_sizes] * n_stages if isinstance(kernel_sizes, int) else list(kernel_sizes)
        f_per  = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else list(features_per_stage)
        n_per  = [n_conv_per_stage] * n_stages if isinstance(n_conv_per_stage, int) else list(n_conv_per_stage)
        st_per = [strides] * n_stages if isinstance(strides, int) else list(strides)
        st_per = [maybe_convert_scalar_to_list(conv_op, s) for s in st_per]

        assert len(ks_per) == len(f_per) == len(n_per) == len(st_per) == n_stages
        if isinstance(kernel_sizes, int):
            self.kernel_sizes_per_stage = [kernel_sizes] * n_stages
        else:
            self.kernel_sizes_per_stage = list(kernel_sizes)
        assert len(self.kernel_sizes_per_stage) == n_stages, \
            f"kernel_sizes_per_stage length ({len(self.kernel_sizes_per_stage)}) mismatch with n_stages ({n_stages})"

        enc_stages, se_blocks = [], []
        inp_ch = input_channels
        for s in range(n_stages):
            # (optional) pooling path
            stage_ops, stage_stride = [], st_per[s]
            if pool in ('max', 'avg'):
                if any(st > 1 for st in st_per[s]):
                    stage_ops.append(get_matching_pool_op(conv_op, pool_type=pool)(
                        kernel_size=st_per[s], stride=st_per[s]
                    ))
                stage_stride = 1

            stage_ops.append(StackedConvBlocks(
                n_per[s], conv_op, inp_ch, f_per[s], ks_per[s], stage_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            ))
            enc_stages.append(nn.Sequential(*stage_ops))

            se_blocks.append(SEBlock3D(
                channels=f_per[s], reduction_ratio=se_reduction_ratio, conv_op=conv_op
            ))
            inp_ch = f_per[s]

        self.stages = nn.Sequential(*enc_stages)
        self.se_blocks = nn.ModuleList(se_blocks)
        self.output_channels = f_per
        self.strides_for_stages = st_per
        self.return_skips = return_skips

    def forward(self, x: torch.Tensor):
        skips, feat = [], x
        for i, stage in enumerate(self.stages):
            feat = stage(feat)
            feat = self.se_blocks[i](feat)   # encoder-side SE refinement
            if self.return_skips:
                skips.append(feat)
        return skips if self.return_skips else feat

    def compute_conv_feature_map_size(self, input_size):
        out = np.int64(0)
        cur = input_size
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        out += self.stages[s][-1].compute_conv_feature_map_size(cur)
            else:
                out += self.stages[s].compute_conv_feature_map_size(cur)
            cur = [i // j for i, j in zip(cur, self.strides_for_stages[s])]
        return out

class UNetDecoder_se(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder_se,
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool,
                 se_reduction_ratio: int = 16,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_enc = len(encoder.output_channels)
        n_dec = n_enc - 1
        n_per = [n_conv_per_stage_decoder] * n_dec if isinstance(n_conv_per_stage_decoder, int) \
            else list(n_conv_per_stage_decoder)
        assert len(n_per) == n_dec

        tconv = get_matching_convtransp(encoder.conv_op)

        cblocks, tconvs, segs, gates = [], [], [], []
        for i in range(n_dec):
            below = encoder.output_channels[-(i + 1)]       # channels from lower level (input of upconv)
            skipc = encoder.output_channels[-(i + 2)]       # channels of matching skip
            stride = encoder.strides_for_stages[-(i + 1)]   # use encoder stride to invert size

            tconvs.append(tconv(below, skipc, stride, stride, bias=(encoder.conv_bias if conv_bias is None else conv_bias)))
            gates.append(SEBlock3D(skipc, se_reduction_ratio, encoder.conv_op))
            cblocks.append(StackedConvBlocks(
                n_per[i], encoder.conv_op, input_channels=2 * skipc, output_channels=skipc,
                kernel_size=encoder.kernel_sizes_per_stage[-(i + 2)], initial_stride=1,
                conv_bias=(encoder.conv_bias if conv_bias is None else conv_bias),
                norm_op=(encoder.norm_op if norm_op is None else norm_op),
                norm_op_kwargs=(encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs),
                dropout_op=(encoder.dropout_op if dropout_op is None else dropout_op),
                dropout_op_kwargs=(encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs),
                nonlin=(encoder.nonlin if nonlin is None else nonlin),
                nonlin_kwargs=(encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs),
                nonlin_first=nonlin_first
            ))
            segs.append(encoder.conv_op(skipc, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(cblocks)
        self.transpconvs = nn.ModuleList(tconvs)
        self.seg_layers = nn.ModuleList(segs)
        self.se_gates = nn.ModuleList(gates)

    def forward(self, skips: List[torch.Tensor]):
        lres = skips[-1]
        seg_outs = []

        for i in range(len(self.stages)):
            x_up = self.transpconvs[i](lres)
            skip = skips[-(i + 2)]
            att = self.se_gates[i](x=skip, g=x_up)
            fused_skip = skip + att
            x = torch.cat([x_up, fused_skip], dim=1)
            x = self.stages[i](x)

            if self.deep_supervision:
                seg_outs.append(self.seg_layers[i](x))
            elif i == len(self.stages) - 1:
                seg_outs.append(self.seg_layers[-1](x))

            lres = x

        seg_outs = seg_outs[::-1]
        return seg_outs[0] if not self.deep_supervision else seg_outs

    def compute_conv_feature_map_size(self, input_size):
        # same logic as before; ASPP is accounted in the wrapper
        skip_sizes, cur = [], input_size
        for s in range(len(self.encoder.strides_for_stages) - 1):
            skip_sizes.append([i // j for i, j in zip(cur, self.encoder.strides_for_stages[s])])
            cur = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)
        out = np.int64(0)
        for s in range(len(self.stages)):
            out += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            out += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)  # trans conv
            if self.deep_supervision or (s == len(self.stages) - 1):
                out += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return out


class PlainConvUNet_se_bottleneck(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 se_reduction_ratio: int = 16,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 aspp_dilations: Tuple[int, ...] = (6, 12, 18)):
        super().__init__()

        n_per = [n_conv_per_stage] * n_stages if isinstance(n_conv_per_stage, int) else list(n_conv_per_stage)
        n_per_dec = [n_conv_per_stage_decoder] * (n_stages - 1) if isinstance(n_conv_per_stage_decoder, int) \
            else list(n_conv_per_stage_decoder)

        # Encoder
        self.encoder = PlainConvEncoder_se(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_per, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first,
            se_reduction_ratio=se_reduction_ratio
        )

        # --- NEW: ASPP bridge at the bottleneck (keeps channels the same by default) ---
        bottleneck_ch = self.encoder.output_channels[-1]
        self.aspp = ASPP(
            conv_op=self.encoder.conv_op,
            in_channels=bottleneck_ch,
            out_channels=bottleneck_ch,            # keep C the same so decoder wiring is unchanged
            dilation_rates=aspp_dilations,
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )

        # Decoder
        self.decoder = UNetDecoder_se(
            self.encoder, num_classes, n_per_dec, deep_supervision,
            se_reduction_ratio=se_reduction_ratio,
            nonlin_first=nonlin_first,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        # replace bottleneck skip with ASPP-processed feature
        skips = [*skips[:-1], self.aspp(skips[-1])]
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        enc = self.encoder.compute_conv_feature_map_size(input_size)
        cur = input_size
        for s in self.encoder.strides_for_stages:
            cur = [i // j for i, j in zip(cur, s)]
        aspp = np.prod([self.encoder.output_channels[-1], *cur], dtype=np.int64)
        dec = self.decoder.compute_conv_feature_map_size(input_size)
        return enc + aspp + dec

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

