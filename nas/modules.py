from typing_extensions import deprecated
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from ConfigSpace import ConfigurationSpace, Integer, Categorical, OrdinalHyperparameter

import naslib.search_spaces.core.primitives as nas_modules
from naslib.search_spaces.core.graph import EdgeData


class ZeroBlock(nas_modules.Zero):
    """Zero block with support for change of number of channels"""

    def forward(
        self, x: torch.Tensor, edge_data: Optional[EdgeData] = None
    ) -> torch.Tensor:
        zeros_shape = list(x.shape)
        zeros_shape[1] = self.C_out  # type: ignore
        return torch.zeros(zeros_shape)


class IdentityBlock(nas_modules.Identity):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """Identity block that projects input to output according to number of channels

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__(**kwargs)
        # Identity uses 1x1 convolution in case of change of channels
        # not as beautiful, but easy to implement
        self.operation = (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, edge_data: Optional[EdgeData] = None
    ) -> torch.Tensor:
        return self.operation(x)


class MaxPoolingBlock(nas_modules.AbstractPrimitive):
    def __init__(self, kernel_size: int, stride: int):
        """Max-pooling block

        Args:
            kernel_size (int): Kernel size of pooling operator
            stride (int): Stride of pooling operator
        """
        super().__init__(locals())
        self.maxpool = nn.MaxPool2d(kernel_size, stride)

    def forward(
        self, x: torch.Tensor, edge_data: Optional[EdgeData] = None
    ) -> torch.Tensor:
        return self.maxpool(x)

    def get_embedded_ops(self):
        return None


class Block(nas_modules.AbstractPrimitive, ABC):
    block_name = "block"

    def __init__(self):
        super().__init__(locals())

        self.layers = []
        self.block = None

    @abstractmethod
    def build_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_BN: bool,
        n_blocks: int,
    ) -> tuple[nn.Module, list]:
        """Constructs building block (with respect to the type) with provided parameters

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolutional kernel
            stride (int): Stride of convolution
            padding (int): Spatial padding that is added to feature map
            use_BN (bool): If true batch normalization is used else not
            n_blocks (int): Number of sequential blocks (each formed of same layers)

        Returns:
            tuple[nn.Module, list]: All sublayers in an sequential module and in a list
        """
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, edge_data: Optional[EdgeData] = None
    ) -> torch.Tensor:
        """Forward pass through module

        Args:
            x (torch.Tensor): input tensor with shape B x C_in x H x W
            edge_data: unused

        Returns:
            torch.Tensor: output tensor with shape B x C_out x H x W
        """
        pass

    def get_embedded_ops(self):
        return self.layers

    @staticmethod
    @deprecated("not used anymore")
    def get_config_space(seed: int = 1) -> ConfigurationSpace:
        """Getter for config space of building block

        Args:
            seed (int, optional): Seed for ConfigurationSpace initialization. Defaults to 1.

        Returns:
            ConfigurationSpace: Config space with hyperparameters for building block
        """
        cs = ConfigurationSpace(seed=seed)
        n_channels = Integer("out_channels", (8, 256), default=8, log=True)
        use_bn = Categorical("use_BN", (True, False), default=True)
        kernel_size = OrdinalHyperparameter(
            "kernel_size", [3, 5, 7, 9], default_value=3
        )
        cs.add([n_channels, use_bn, kernel_size])
        return cs

    @classmethod
    @deprecated("not used anymore")
    def add_config_space_to_model(
        cls, cs_net: ConfigurationSpace, block_prefix: str, seed: int = 0
    ) -> ConfigurationSpace:
        """Adds the config space of the current module to the config space of the cs_net

        Args:
            cs_net (ConfigurationSpace): The network config space to which the current module should be added
            block_prefix (str): To which block layer the current block belongs
            seed (int, optional): Seed. Defaults to 0.

        Returns:
            ConfigurationSpace: Joint configuration space
        """
        block_choice = f"{block_prefix}:__choice__"
        if block_choice not in cs_net or not hasattr(cs_net[block_choice], "compare"):
            parent_hyperparameter = None
        else:
            parent_hyperparameter = {
                "parent": cs_net[f"{block_prefix}:__choice__"],
                "value": f"{cls.block_name}",
            }
        cs_net.add_configuration_space(
            prefix=f"{block_prefix}:{cls.block_name}",
            configuration_space=cls.get_config_space(seed=seed),
            parent_hyperparameter=parent_hyperparameter,
        )
        return cs_net

    def __init_subclass__(cls):
        """Checks if 'block_name' attribute is overwritten by subclasses

        Raises:
            NotImplementedError: 'block_name' attribute was not overwritten by subclass
        """
        if not any(
            "block_name" in base.__dict__ for base in cls.__mro__ if base is not Block
        ):
            raise NotImplementedError(
                f"Attribute 'block_name' has not been overwritten in class '{cls.__name__}'"
            )


class FinalBlock(Block):
    block_name = "linear"

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        intermediate_features: list[int],
        out_features: int,
        dropout_rate: float,
    ):
        """Block for final layers of CNN. It includes global average pooling followed by up to k linear layers

        Args:
            num_layers (int): Number of linear layers
            in_channels (int): Number of channels of last convolutional layer for each linear layer
            intermediate_features (list[int]): Number of output_features for each linear layer
            out_channels (int): Number of output channels of last mapping layer (e.g. number of classes)
            dropout_rate (float): Dropout rate for the linear layers
        """
        super().__init__()

        assert num_layers == len(
            intermediate_features
        ), "length of in_channels must match num_layers"
        self.num_layers = num_layers

        assert (
            0.0 <= dropout_rate < 1.0
        ), "Dropout rate has to be probability in range [0, 1)"
        self.dropout_rate = dropout_rate

        self.block, self.layers = self.build_block(
            in_channels, intermediate_features, out_features
        )

    def build_block(
        self, in_channels: int, intermediate_features: list[int], out_features: int
    ) -> tuple[nn.Module, list]:
        """Differs from abstract class signature

        Args:
            in_channels (int): Number of input channels (from last convolutional layer)
            intermediate_features (list[int]): Number of output features for each linear layer
            out_features (int): Number of output_values of last linear mapping

        Returns:
            tuple[nn.Module, list]: All sublayers in an sequential module and in a list
        """
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]

        input_sizes: list[int] = [in_channels] + intermediate_features[:-1]
        output_sizes: list[int] = intermediate_features

        for input_size, output_size in zip(input_sizes, output_sizes):
            layers += [
                nn.Linear(input_size, output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate),
            ]
        layers.append(nn.Linear(intermediate_features[-1], out_features))
        return nn.Sequential(*layers), layers

    def forward(self, x: torch.Tensor, edge_data=None) -> torch.Tensor:
        return self.block(x)

    @staticmethod
    def get_config_space(seed: int = 1) -> ConfigurationSpace:
        """Empty config space because hyperparameters are already included in model config space

        Args:
            seed (int, optional): Seed for ConfigurationSpace initialization. Defaults to 1.

        Returns:
            ConfigurationSpace: Empty config space
        """
        cs = ConfigurationSpace(seed=seed)
        return cs


class ConvBlock(Block):
    block_name = "conv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_BN: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        n_blocks = kwargs.get("n_blocks", 2)

        # if no padding is provided it preserves spatial dimensions
        padding = kwargs.get("padding", int(np.ceil((kernel_size - 1) / 2)))

        # so it can be used as module inside resnet and resnetbottleneck blocks
        self.apply_final_activation = kwargs.get("apply_final_activation", True)

        self.block, self.layers = self.build_block(
            in_channels, out_channels, kernel_size, stride, padding, use_BN, n_blocks
        )

    def build_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_BN: bool,
        n_blocks: int,
    ) -> tuple[nn.Module, list]:
        layers = []

        for i in range(n_blocks):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            if use_BN:
                layers.append(nn.BatchNorm2d(out_channels))

            if i < n_blocks - 1 or self.apply_final_activation:
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        return nn.Sequential(*layers), layers

    def forward(self, x, edge_data=None):
        return self.block(x)


class ResNetConvBlock(Block):
    block_name = "resnet_conv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_BN: bool = True,
    ) -> None:
        super().__init__()
        padding = int(np.ceil((kernel_size - 1) / 2))

        self.preprocessing_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # preprocessing layer maps already in to out channels
        self.block, self.layers = self.build_block(
            out_channels, out_channels, kernel_size, stride, padding, use_BN, n_blocks=2
        )
        self.relu = nn.ReLU()

    def build_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_BN: bool,
        n_blocks: int,
    ) -> tuple[nn.Module, list]:
        block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            use_BN,
            n_blocks=n_blocks,
            padding=padding,
            apply_final_activation=False,
        )
        return block.block, block.layers

    def forward(self, x, edge_data=None):
        x = self.preprocessing_layer(x)
        return self.relu(x + self.block(x))


class ResNetBottleneckConvBlock(Block):
    block_name = "resnet_bottleneck_conv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_BN: bool = True,
    ) -> None:
        super().__init__()
        padding = int(np.ceil((kernel_size - 1) / 2))

        self.preprocessing_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # preprocessing layer maps already in to out channels
        self.block, self.layers = self.build_block(
            out_channels, out_channels, kernel_size, stride, padding, use_BN, n_blocks=1
        )
        self.relu = nn.ReLU()

    def build_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_BN: bool,
        n_blocks: int,
    ) -> tuple[nn.Module, list]:
        intermediate_channels = out_channels // 4
        layers = []

        for i in range(n_blocks):
            layers.append(
                ConvBlock(
                    in_channels,
                    intermediate_channels,
                    kernel_size=1,
                    stride=1,
                    use_BN=use_BN,
                    n_blocks=1,
                    padding=0,
                )
            )

            layers.append(
                ConvBlock(
                    intermediate_channels,
                    intermediate_channels,
                    kernel_size,
                    stride,
                    use_BN=use_BN,
                    n_blocks=1,
                    padding=padding,
                )
            )

            apply_final_activation = False
            if i < n_blocks - 1:
                apply_final_activation = True
            layers.append(
                ConvBlock(
                    intermediate_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    use_BN=use_BN,
                    n_blocks=1,
                    padding=0,
                    apply_final_activation=apply_final_activation,
                )
            )

            in_channels = out_channels

        return nn.Sequential(*layers), layers

    def forward(self, x, edge_data=None):
        x = self.preprocessing_layer(x)
        return self.relu(x + self.block(x))
