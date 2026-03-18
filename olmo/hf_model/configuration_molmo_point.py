"""
Molmo2 configuration
"""

from typing import Optional

from transformers import PretrainedConfig, LogitsProcessor
from transformers.utils import logging

from .configuration_molmo2 import Molmo2TextConfig, Molmo2VitConfig, \
    Molmo2AdapterConfig

logger = logging.get_logger(__name__)


class MolmoPointAdapterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of Molmo2Adapter. With Molmo2VitConfig,
    It is used to instantiate an Molmo2VisionBackbone according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import Molmo2VitConfig, Molmo2AdapterConfig, Molmo2VisionBackbone

    >>> # Initializing a Molmo2VitConfig and a Molmo2AdapterConfig
    >>> vit_config = Molmo2VitConfig()
    >>> adapter_config = MolmoPoolingConfig()

    >>> # Initializing a Molmo2VisionBackbone (with random weights)
    >>> model = Molmo2VisionBackbone(vit_config, adapter_config)

    >>> # Accessing the model configuration
    >>> vit_configuration = model.vit_config
    >>> adapter_configuration = model.adapter_config
    ```"""

    model_type = "molmo_point"
    base_config_key = "adapter_config"

    def __init__(
        self,
        vit_layers: tuple = (-3, -9),
        pooling_attention_mask: bool = False,
        hidden_size: int = 1152,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        text_hidden_size: int = 3584,
        image_feature_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attn_implementation: str = "eager",
        positional_embeddings: int = 16,
        attention_pooling_out_layer: bool = False,
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.vit_layers = vit_layers
        self.pooling_attention_mask = pooling_attention_mask
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_feature_dropout = image_feature_dropout
        self.initializer_range = initializer_range
        self.positional_embeddings = positional_embeddings
        self.attention_pooling_out_layer = attention_pooling_out_layer


class MolmoPointConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoPointForConditionalGeneration`].
    It is used to instantiate an Molmo2 model according to the specified arguments, defining the model architecture.

    Example:

    ```python
    >>> from transformers import Molmo2Config, Molmo2VitConfig, Molmo2AdapterConfig, Molmo2TextConfig

    >>> # Initializing a Molmo2VitConfig
    >>> vit_config = Molmo2VitConfig()

    >>> # Initializing a Molmo2AdapterConfig
    >>> adapter_config = MolmoPointAdapterConfig()

    >>> # Initializing a Molmo2TextConfig
    >>> text_config = Molmo2TextConfig()

    >>> # Initializing a Molmo2Config
    >>> configuration = MolmoPointConfig(
    >>>     vit_config=vit_config,
    >>>     adapter_config=adapter_config,
    >>>     text_config=text_config,
    >>>     image_start_token_id=151936,
    >>>     image_end_token_id=151937,
    >>>     image_patch_id=151938,
    >>>     image_col_id=151939,
    >>>     low_res_image_start_token_id=151940,
    >>>     image_low_res_id=151942,
    >>>     frame_start_token_id=151943,
    >>>     frame_end_token_id=151944,
    >>> )

    >>> # Initializing a model
    >>> model = MolmoPointForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmo_point"
    sub_configs = {
        "text_config": Molmo2TextConfig,
        "vit_config": Molmo2VitConfig,
        "adapter_config": MolmoPointAdapterConfig,
    }

    def __init__(
        self,
        vit_config: Molmo2VitConfig = None,
        adapter_config: MolmoPointAdapterConfig = None,
        text_config: Molmo2TextConfig = None,
        image_start_token_id: int = None,
        low_res_image_start_token_id: int = None,
        image_end_token_id: int = None,
        image_patch_id: int = None,
        image_non_indexable_patch_id: int = None,
        image_col_id: int = None,
        frame_start_token_id: int = None,
        frame_end_token_id: int = None,
        patch_token_id: int = None,
        subpatch_token_id: int = None,
        location_token_id: int = None,
        use_frame_special_tokens: bool = True,
        initializer_range: float = 0.02,

        # point config
        patch_location: Optional[str]="3x3",
        no_more_points_class: bool=False,
        patch_embed_dim: int=256,
        patch_embedding_kind: str="linear",
        embed_selected_vit_patch: Optional[str]="linear",
        embed_location: bool=False,
        layer_norm_x: bool=True,
        norm_logits: bool=True,
        # FIXME figure out how infernce params work
        mask_patches: Optional[str]="always",
        mask_subpatches: str="inference",
        mask_repeats: Optional[str]="inference",
        token_prediction_rotary: bool=True,
        token_prediction_rotary_theta: Optional[float]=50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vit_config is None:
            self.vit_config = Molmo2VitConfig()
        elif isinstance(vit_config, dict):
            self.vit_config = Molmo2VitConfig(**vit_config)
        else:
            self.vit_config = vit_config
        if adapter_config is None:
            self.adapter_config = Molmo2AdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = Molmo2AdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        if text_config is None:
            self.text_config = Molmo2TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Molmo2TextConfig(**text_config)
        else:
            self.text_config = text_config
        self.image_start_token_id = image_start_token_id
        self.low_res_image_start_token_id = low_res_image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_high_res_id = image_patch_id
        self.image_non_indexable_patch_id = image_non_indexable_patch_id
        self.image_patch_id = image_patch_id
        self.image_col_id = image_col_id
        self.frame_start_token_id = frame_start_token_id
        self.frame_end_token_id = frame_end_token_id
        self.patch_token_id = patch_token_id
        self.subpatch_token_id = subpatch_token_id
        self.location_token_id = location_token_id
        self.use_frame_special_tokens = use_frame_special_tokens
        self.initializer_range = initializer_range
        self.patch_location = patch_location
        self.no_more_points_class = no_more_points_class
        self.patch_embed_dim = patch_embed_dim
        self.patch_embedding_kind = patch_embedding_kind
        self.embed_selected_vit_patch = embed_selected_vit_patch
        self.embed_location = embed_location
        self.layer_norm_x = layer_norm_x
        self.norm_logits = norm_logits
        self.mask_patches = mask_patches
        self.mask_subpatches = mask_subpatches
        self.mask_repeats = mask_repeats
        self.token_prediction_rotary = token_prediction_rotary
        self.token_prediction_rotary_theta = token_prediction_rotary_theta

    @property
    def image_num_patch(self):
        assert self.vit_config is not None
        return self.vit_config.image_num_patch
    
    @property
    def num_attention_heads(self):
        return self.text_config.num_attention_heads
    
    @property
    def num_key_value_heads(self):
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.text_config.head_dim

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers
    
    @property
    def hidden_size(self):
        return self.text_config.hidden_size
    
    @property
    def vocab_size(self):
        return self.text_config.vocab_size
    
    @property
    def max_position_embeddings(self):
        return self.text_config.max_position_embeddings


MolmoPointAdapterConfig.register_for_auto_class()
MolmoPointConfig.register_for_auto_class()