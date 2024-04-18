""" AdaptFormer model configuration"""

from transformers import PretrainedConfig


class AdaptFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AdaptFormerForChangeDetection`].
    It is used to instantiate an AdaptFormer model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the AdaptFormer
    [deepang/adaptformer-LEVIR-CD](https://huggingface.co/deepang/adaptformer-LEVIR-CD)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_classes (`int`, *optional*, defaults to 2):
            The number of classes.
        embed_dims (`List[int]`, *optional*, defaults to `[64, 128, 256]`):
            Dimension of each of the encoder blocks.
        num_heads (`List[int]`, *optional*, defaults to `[1, 2, 4]`):
            Number of attention heads for each attention layer in each block of the encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3]`):
            The number of layers in each encoder block.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.
        semantic_loss_weight (`List[float]`, *optional*, defaults to `[0, 0, 0.8, 1]`):
            The weight of the semantic segmentation loss.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import AutoModel, AutoConfig

    >>> # Initializing a AdaptFormer
    >>> configuration = AutoConfig.from_pretrained("deepang/adaptformer-LEVIR-CD")

    >>> # Initializing a model from the deepang/adaptformer-LEVIR-CD style configuration
    >>> model = AutoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "adaptformer"

    def __init__(
        self,
        num_channels=3,
        num_classes=2,
        embed_dims=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        depths=[3, 3, 3],
        semantic_loss_ignore_index=255,
        semantic_loss_weight=[0, 0, 0.5, 1],
        initializer_range=0.02,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.depths = depths
        self.num_classes = num_classes
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.semantic_loss_weight = semantic_loss_weight
        self.initializer_range = initializer_range

        super().__init__(**kwargs)
