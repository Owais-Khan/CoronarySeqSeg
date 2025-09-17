import torch
import torch.nn as nn
from functools import partial
import os
# import psutil # Not used in core model
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
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
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

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)



class SkipAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
        )
    def forward(self, x):
        return x + self.net(x)


# --- SEBlock3D Module ---
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16, conv_op=nn.Conv3d):
        super().__init__()
        red = max(1, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.channel_exc = nn.Sequential(
            nn.Linear(channels, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, g=None):
        # Optional cross‐attn or gating on x+g here…
        b, c, d, h, w = x.shape
        # Channel attention
        chn = self.pool(x).view(b, c)
        chn = self.channel_exc(chn).view(b, c, 1, 1, 1)
        x_ch = x * chn

        # Spatial attention
        spa = x_ch.mean(dim=1, keepdim=True)                # [b,1,d,h,w]
        spa_mask = self.spatial_gate(spa)                   # [b,1,d,h,w]
        return x_ch * spa_mask



# --- End of SEBlock3D Module ---

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
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class PlainConvEncoder_sefc(nn.Module):
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


class UNetDecoder_sefc(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder_sefc,
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
                 conv_bias: bool = None
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder_list = [n_conv_per_stage_decoder] * (n_stages_encoder - 1)
        else:
            n_conv_per_stage_decoder_list = list(n_conv_per_stage_decoder)
        assert len(n_conv_per_stage_decoder_list) == n_stages_encoder - 1, \
            f"n_conv_per_stage_decoder length ({len(n_conv_per_stage_decoder_list)}) mismatch with expected ({n_stages_encoder - 1})"

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_op = encoder.conv_op

        conv_bias_resolved = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op_resolved = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs_resolved = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op_resolved = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs_resolved = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin_resolved = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs_resolved = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        decoder_conv_stages = []
        transpconvs = []
        seg_layers = []
        self.se_gates = nn.ModuleList()
        # New module list for inter-decoder SE blocks
        self.decoder_se_blocks = nn.ModuleList()

        for s_idx_loop in range(n_stages_encoder - 1):
            s_map_to_paper = s_idx_loop + 1

            input_features_below = encoder.output_channels[-s_map_to_paper]
            input_features_skip = encoder.output_channels[-(s_map_to_paper + 1)]

            stride_for_transpconv = encoder.strides_for_stages[-s_map_to_paper]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias_resolved
            ))

            # SE block for the attention-based skip path (Attention Gate)
            self.se_gates.append(SEBlock3D(
                channels=input_features_skip,
                reduction_ratio=se_reduction_ratio,
                conv_op=encoder.conv_op
            ))

            # The input to the decoder's conv block is the concatenation of the upsampled features (C channels)
            # and the fused skip connection (also C channels), resulting in 2*C channels.
            decoder_conv_stages.append(StackedConvBlocks(
                num_convs=n_conv_per_stage_decoder_list[s_idx_loop],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes_per_stage[-(s_map_to_paper + 1)],
                initial_stride=1,
                conv_bias=conv_bias_resolved,
                norm_op=norm_op_resolved, norm_op_kwargs=norm_op_kwargs_resolved,
                dropout_op=dropout_op_resolved, dropout_op_kwargs=dropout_op_kwargs_resolved,
                nonlin=nonlin_resolved, nonlin_kwargs=nonlin_kwargs_resolved,
                nonlin_first=nonlin_first
            ))

            # Add an SE block to refine the output of the current decoder stage
            self.decoder_se_blocks.append(SEBlock3D(
                channels=input_features_skip,
                reduction_ratio=se_reduction_ratio,
                conv_op=encoder.conv_op
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(decoder_conv_stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor]):
        lres_input = skips[-1]
        seg_outputs = []

        for s_idx in range(len(self.stages)):
            x_up = self.transpconvs[s_idx](lres_input)
            skip_from_encoder = skips[-(s_idx + 2)]
            attended_skip = self.se_gates[s_idx](x=skip_from_encoder, g=x_up)
            fused_skip = skip_from_encoder + attended_skip
            x_concat = torch.cat((x_up, fused_skip), dim=1)

            current_stage_output = self.stages[s_idx](x_concat)

            # Apply SE block to the decoder stage output before passing it to the next stage
            current_stage_output = self.decoder_se_blocks[s_idx](current_stage_output)

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
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class PlainConvUNet_sefc(nn.Module):
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
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage_list = [n_conv_per_stage] * n_stages
        else:
            n_conv_per_stage_list = list(n_conv_per_stage)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder_list = [n_conv_per_stage_decoder] * (n_stages - 1)
        else:
            n_conv_per_stage_decoder_list = list(n_conv_per_stage_decoder)

        self.encoder = PlainConvEncoder_sefc(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage_list, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first,
            se_reduction_ratio=se_reduction_ratio
        )
        self.decoder = UNetDecoder_sefc(
            self.encoder, num_classes, n_conv_per_stage_decoder_list, deep_supervision,
            se_reduction_ratio=se_reduction_ratio,
            nonlin_first=nonlin_first,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias
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

    @staticmethod
    def initialize(module):
        # Placeholder for He initialization, assuming InitWeights_He exists elsewhere
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
