import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.new_architectures.unet_se import PlainConvUNet_se
from nnunetv2.new_architectures.unet_se_fully_connected import PlainConvUNet_sefc
from nnunetv2.new_architectures.unet_ConvLSTM import PlainConvUNet_ConvLSTM
from batchgenerators.utilities.file_and_folder_operations import join


def import_new_architecture(model: str):
    if model == 'unet_se':
        network = PlainConvUNet_se
    elif model == 'unet_se_fully_connected':
        network = PlainConvUNet_sefc
    elif model == 'unet_ConvLSTM':
        network = PlainConvUNet_ConvLSTM
    else:
        print('[Err] Architecture unknown falling back to nnunet')
        network = None
    return network

def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    model = arch_class_name.rsplit(".", 2)[-2]
    nw_class = import_new_architecture(model)


    # sometimes things move around, this makes it so that we can at least recover some of that
    '''if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        print(nw_class)
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision'''

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_class_name="nnunetv2.new_architectures.unet_ConvLSTM.PlainConvUNet_ConvLSTM",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320,
                        320
                    ],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [
                        [
                            1,
                            3,
                            3
                        ],
                        [
                            1,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
            "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            1,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ]
                    ],
            "n_conv_per_stage": [
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
            "n_conv_per_stage_decoder": [
                        2,
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},

            "se_reduction_ratio": 16,
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=3,
        allow_init=True,
        deep_supervision=False,
    )
    data = torch.rand(1, 1,28,128,128)
    target = torch.rand(size=(1, 1,28,128,128))
    outputs = model(data) # this should be a list of torch.Tensor