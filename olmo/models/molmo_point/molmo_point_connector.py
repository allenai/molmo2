from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard
from torch.nn import init

from olmo.config import BaseConfig
from olmo.nn.image_vit import ViTMultiHeadDotProductAttention
from olmo.nn.llm import LlmConfig, LayerNormBase, Activation
from olmo.nn.vision_backbone import ImageProjectorMLP
from olmo.preprocessing.image_preprocessor import ImagePreprocessor


@dataclass
class ConnectorConfig(BaseConfig):
    """The Image/Language connector"""

    pooling_attention_mask: bool = True
    """Use an attention mask when pooling instead setting masked embeddings to 0"""

    image_projector: str = "mlp"
    """Layer to project pooled image features to the LLM embedding space"""

    pooling_out_layer: bool = True
    """Include the out layers for the Attention pooler
    
    This layer is redundant since its immediately followed by the MLP linear layer, so 
    it can be removed. 
    """

    vit_layers: Tuple = (-1,)
    """What layers to use from the VIT"""

    skip_unused_layers: bool = True
    """Don't load layers we don't need from the ViT"""

    positional_embeddings: Optional[int] = 9
    """Add positional embeddings to the pooled image features"""

    connector_activation_checkpointing: bool = True
    """Allow activation checkpoint on the connector components"""

    compile_connector: Optional[str] = None
    """Compile the connector"""

    normalize_on_gpu: bool = False
    """Run image normalization on the GPU
    
    Doing this will allow image loading to keep the images in uint8 which will reduce 
    RAM/shared memory usage significantly
    """

    @classmethod
    def update_legacy_settings(cls, config):
        if config.positional_embeddings is True:
            if "share_pooler" in config:
                config.positional_embeddings = 16
            else:
                config.positional_embeddings = 9
        if "query" in config:
            assert config.pop("query") == "mean"
        if "shared" in config:
            assert config.pop("shared")
        for k in ["share_pooler", "image_padding_embed", "share_projector", "image_feature_dropout"]:
            if k in config:
                assert not config.pop(k)
        return config

    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]

    def build_preprocessor(self, vit):
        return ImagePreprocessor(
            normalize=vit.normalize,
            resize=vit.resize_mode,
            pad_value=vit.pad_value,
            image_patch_size=vit.image_patch_size,
            base_image_input_size=vit.image_default_input_size,
            normalize_on_gpu=self.normalize_on_gpu,
            use_image_mask=False
        )

    def build(self, llm_config, vit_config, device):
        return MolmoPointConnector(self, llm_config, vit_config, device)


class AddPosEmbed(nn.Module):

    def __init__(self, in_features: int, n_pos: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([n_pos, in_features]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias[None, :input.shape[-2], :]


class MolmoPointConnector(nn.Module):
    def __init__(self, config: ConnectorConfig, llm_config: LlmConfig, vit_config, device):
        super().__init__()
        self.config = config
        self.vit_config = vit_config
        self.image_preprocessor = config.build_preprocessor(vit_config)
        vit_cfg = self.vit_config
        self.n_vit_layers = len(config.vit_layers)
        pool_dim = vit_cfg.image_emb_dim * self.n_vit_layers

        self.norm = None

        if self.config.image_projector == "mlp":
            pool_out_dim: int = self.vit_config.image_emb_dim
            self.image_projector = ImageProjectorMLP(llm_config, pool_out_dim, device=device)
        else:
            raise NotImplementedError(self.config.image_projector)

        if not config.pooling_out_layer:
            pool_out_dim = -1
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            vit_cfg, input_dim=pool_dim, out_dim=pool_out_dim,)

        if self.config.positional_embeddings:
            self.positional_embeddings = AddPosEmbed(pool_dim, self.config.positional_embeddings)
        else:
            self.positional_embeddings = None

    def apply_compile(self, **kwargs):
        if self.config.compile_connector:
            if self.config.compile_connector == "dynamic":
                connect_kwargs = dict(kwargs, dynamic=True)
            elif self.config.compile_connector == "default":
                connect_kwargs = kwargs
            else:
                raise NotImplementedError(self.config.compile_connector)
            self.image_pooling_2d.compile(**connect_kwargs)
            self.image_projector.compile(**connect_kwargs)

    def apply_activation_checkpointing(self):
        if self.config.connector_activation_checkpointing:
            self.image_projector = checkpoint_wrapper(self.image_projector)
            self.image_pooling_2d = checkpoint_wrapper(self.image_pooling_2d)

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        if self.config.image_projector == "mlp_merge":
            modules = [self.image_projector, self.image_pooling_2d]
            if self.norm is not None:
                modules.append(self.norm)
            if self.positional_embeddings is not None:
                modules.append(self.positional_embeddings)
            fully_shard(modules, **fully_shard_kwargs)
        else:
            if self.positional_embeddings is not None:
                fully_shard([self.positional_embeddings, self.image_projector], **fully_shard_kwargs)
            else:
                fully_shard(self.image_projector, **fully_shard_kwargs)
            if self.norm is not None:
                fully_shard([self.norm, self.image_pooling_2d], **fully_shard_kwargs)
            else:
                fully_shard(self.image_pooling_2d, **fully_shard_kwargs)
        fully_shard(self, **fully_shard_kwargs)

    def reset_parameters(self):
        self.image_pooling_2d.reset_parameters()
        if isinstance(self.image_projector, nn.Linear):
            if self.image_projector.bias is not None:
                init.constant_(self.image_projector.bias, 0)
            init.normal_(self.image_projector.weight, std=0.02)
        else:
            self.image_projector.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.positional_embeddings is not None:
            init.zeros_(self.positional_embeddings.bias)

    def __call__(self, to_pool, to_pool_mask):
        """
        to_pool: [n_to_pool, pooling_dim, vit_dim]
        to_pool_mask: [n_to_pool, pooling_dim]

        returns:
        pooled_features: [n_to_pool, llm_dim]
        """
        cfg = self.config

        if self.config.positional_embeddings:
            to_pool = self.positional_embeddings(to_pool)

        if self.config.pooling_attention_mask:
            attn_mask = to_pool_mask.reshape([-1, 1, 1, to_pool_mask.shape[-1]])
        else:
            attn_mask = None
            to_pool = to_pool * to_pool_mask.float()[:, :, None]

        denom = to_pool_mask.view(-1, to_pool.shape[-2]).float().sum(-1)
        denom = torch.where(denom == 0, 1, denom)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]

        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = self.image_projector(pooled_features)
        return pooled_features
