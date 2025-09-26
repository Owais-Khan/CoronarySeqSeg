import torch
import torch.nn as nn
from functools import partial
import os
import json
import numpy as np
from typing import Union, Type, List, Tuple
import time

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

import torch
import torch.nn as nn
from functools import partial
import os
import json
import numpy as np
from typing import Union, Type, List, Tuple
import time

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
    get_matching_convtransp,
    convert_conv_op_to_dim
)

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks,ConvDropoutNormReLU
class ASPP(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 in_channels: int,
                 out_channels: int,
                 dilation_rates: Tuple[int, ...] = (6, 12, 18),
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = False):
        """
        Atrous Spatial Pyramid Pooling module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1x1 = conv_op(in_channels, out_channels, kernel_size=1, bias=conv_bias)
        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(
                conv_op(in_channels, out_channels, kernel_size=3, padding=rate,
                        dilation=rate, bias=conv_bias)
            )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1) if conv_op == nn.Conv3d else nn.AdaptiveAvgPool2d(1),
            conv_op(in_channels, out_channels, kernel_size=1, bias=conv_bias)
        )
        total_channels = out_channels * (len(dilation_rates) + 2)
        self.final_conv = conv_op(total_channels, out_channels, kernel_size=1, bias=conv_bias)
        self.norm_nonlin = nn.Sequential()
        if norm_op is not None:
            self.norm_nonlin.append(norm_op(out_channels, **norm_op_kwargs))
        if nonlin is not None:
            self.norm_nonlin.append(nonlin(**nonlin_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        out_1x1 = self.conv_1x1(x)
        atrous_outs = [out_1x1]
        for conv in self.atrous_convs:
            atrous_outs.append(conv(x))
        img_pool_out = self.image_pool(x)
        img_pool_out = F.interpolate(img_pool_out, size=size, mode='trilinear' if len(size) == 3 else 'bilinear',
                                     align_corners=False)
        atrous_outs.append(img_pool_out)
        x = torch.cat(atrous_outs, dim=1)
        x = self.final_conv(x)
        x = self.norm_nonlin(x)
        return x




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


class PlainConvEncoder_ConvLSTM(nn.Module):
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
                 se_reduction_ratio: int = 16
                 ):
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
        self.strides=strides

        if isinstance(kernel_sizes, int):
            self.kernel_sizes_per_stage = [kernel_sizes] * n_stages
        else:
            self.kernel_sizes_per_stage = list(kernel_sizes)
        assert len(self.kernel_sizes_per_stage) == n_stages, \
            f"kernel_sizes_per_stage length ({len(self.kernel_sizes_per_stage)}) mismatch with n_stages ({n_stages})"

        if isinstance(features_per_stage, int):
            features_per_stage_list = [features_per_stage] * n_stages
        else:
            features_per_stage_list = list(features_per_stage)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage_list = [n_conv_per_stage] * n_stages
        else:
            n_conv_per_stage_list = list(n_conv_per_stage)
        if isinstance(strides, int):
            strides_list = [strides] * n_stages
        else:
            strides_list = list(strides)

        processed_strides_list = []
        for s_val in strides_list:
            processed_strides_list.append(maybe_convert_scalar_to_list(conv_op, s_val))
        strides_list = processed_strides_list

        assert len(features_per_stage_list) == n_stages
        assert len(n_conv_per_stage_list) == n_stages
        assert len(strides_list) == n_stages

        encoder_stages_modules = []
        self.se_blocks = nn.ModuleList()  # SE blocks for each encoder stage
        current_input_channels = input_channels
        for s in range(n_stages):
            stage_ops = []
            stage_conv_stride = strides_list[s]

            if pool == 'max' or pool == 'avg':
                if any(st > 1 for st in strides_list[s]):
                    stage_ops.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides_list[s],
                                                                      stride=strides_list[s])
                    )
                stage_conv_stride = 1
            elif pool != 'conv':
                raise RuntimeError(f"Unsupported pool type: {pool}")

            stage_ops.append(StackedConvBlocks(
                n_conv_per_stage_list[s], conv_op, current_input_channels, features_per_stage_list[s],
                self.kernel_sizes_per_stage[s], stage_conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            ))
            encoder_stages_modules.append(nn.Sequential(*stage_ops))

            # Add an SE Block after each stage's convolutions
            self.se_blocks.append(SEBlock3D(
                channels=features_per_stage_list[s],
                reduction_ratio=se_reduction_ratio,
                conv_op=conv_op
            ))

            current_input_channels = features_per_stage_list[s]

        self.stages = nn.Sequential(*encoder_stages_modules)
        self.output_channels = features_per_stage_list
        self.strides_for_stages = strides_list
        self.return_skips = return_skips

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = []
        current_features = x
        for i, stage_module in enumerate(self.stages):
            current_features = stage_module(current_features)
            # Apply SE block to refine features of the current stage
            current_features = self.se_blocks[i](current_features)
            if self.return_skips:
                skips.append(current_features)

        if self.return_skips:
            return skips
        else:
            return current_features

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output



class LightConvLSTMSkip(nn.Module):
    def __init__(self, in_ch, r=4, kernel_size=3):
        super().__init__()
        self.hidden = None
        self.reduced_ch = max(1, in_ch // r)
        padding = kernel_size // 2
        self.reduce = nn.Conv3d(in_ch, self.reduced_ch, 1, bias=False)
        self.dw_conv = nn.Conv3d(self.reduced_ch * 2, self.reduced_ch * 4,
                                 kernel_size, padding=padding,
                                 groups=self.reduced_ch, bias=False)
        self.pw_conv = nn.Conv3d(self.reduced_ch * 4, self.reduced_ch * 4,
                                 1, bias=True)
        self.restore = nn.Conv3d(self.reduced_ch, in_ch, 1, bias=False)

    def forward(self, skip_feat, decoder_feat):
        if decoder_feat.shape[2:] != skip_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, size=skip_feat.shape[2:], mode='trilinear', align_corners=False
            )
        x = self.reduce(skip_feat)
        if self.hidden is None or self.hidden[0].shape[2:] != x.shape[2:]:
            self.hidden = (torch.zeros_like(x), torch.zeros_like(x))
        h, c = self.hidden

        inp = torch.cat([x, self.reduce(decoder_feat)], dim=1)
        gates = self.pw_conv(self.dw_conv(inp))
        i, f, g, o = torch.split(gates, self.reduced_ch, dim=1)

        i = torch.sigmoid(i);
        f = torch.sigmoid(f)
        g = torch.tanh(g);
        o = torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        self.hidden = (new_h.detach(), new_c.detach())

        restored = self.restore(new_h)
        return skip_feat + restored

class PlainConvDecoder_ConvLSTM(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder_ConvLSTM,
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool,
                 se_reduction_ratio: int = 16,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 strides: Union[int, List[int], Tuple[int, ...]]=None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.strides = [maybe_convert_scalar_to_list(nn.Conv3d, i) for i in strides]
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder_list = [n_conv_per_stage_decoder] * (n_stages_encoder - 1)
        else:
            n_conv_per_stage_decoder_list = list(n_conv_per_stage_decoder)

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # Module Lists
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.se_gates = nn.ModuleList()
        self.lstm_skips = nn.ModuleList()

        for s_idx_loop in range(n_stages_encoder - 1):
            input_features_below = encoder.output_channels[-(s_idx_loop + 1)]
            input_features_skip = encoder.output_channels[-(s_idx_loop + 2)]
            stride_for_transpconv = encoder.strides_for_stages[-(s_idx_loop + 1)]

            self.transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))

            self.se_gates.append(SEBlock3D(
                channels=input_features_skip,
                reduction_ratio=se_reduction_ratio,
                conv_op=encoder.conv_op
            ))

            self.lstm_skips.append(LightConvLSTMSkip(in_ch=input_features_skip))

            self.stages.append(StackedConvBlocks(
                num_convs=n_conv_per_stage_decoder_list[s_idx_loop],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,  # Input is still 2*C
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes_per_stage[-(s_idx_loop + 2)],
                initial_stride=1,
                conv_bias=conv_bias, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
            ))

            self.seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

    def forward(self, skips: List[torch.Tensor]):
        lres_input = skips[-1]
        seg_outputs = []

        for s_idx in range(len(self.stages)):
            x_up = self.transpconvs[s_idx](lres_input)
            skip_from_encoder = skips[-(s_idx + 2)]
            enriched_skip = self.lstm_skips[s_idx](skip_from_encoder, x_up)
            attended_skip = self.se_gates[s_idx](x=enriched_skip, g=x_up)
            x_concat = torch.cat((x_up, attended_skip), dim=1)

            current_stage_output = self.stages[s_idx](x_concat)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s_idx](current_stage_output))
            elif s_idx == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](current_stage_output))

            lres_input = current_stage_output

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            if not seg_outputs: raise RuntimeError("No segmentation output was produced.")
            return seg_outputs[0]
        else:
            return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output




class PlainConvUNet_ConvLSTM(nn.Module):
    def __init__(
            self,
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
    ):
        super().__init__()
        self.strides=strides
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage_list = [n_conv_per_stage] * n_stages
        else:
            n_conv_per_stage_list = list(n_conv_per_stage)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder_list = [n_conv_per_stage_decoder] * (n_stages - 1)
        else:
            n_conv_per_stage_decoder_list = list(n_conv_per_stage_decoder)

        self.encoder = PlainConvEncoder_ConvLSTM(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage_list, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first,
            se_reduction_ratio=se_reduction_ratio
        )
        self.decoder = PlainConvDecoder_ConvLSTM(
            self.encoder, num_classes, n_conv_per_stage_decoder_list, deep_supervision,
            se_reduction_ratio=se_reduction_ratio,
            nonlin_first=nonlin_first,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,strides=strides
        )


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )
