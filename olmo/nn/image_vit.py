import logging
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.backends.cuda
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.activations import get_activation

from olmo.config import BaseConfig, StrEnum, D
from olmo.nn.llm import AttentionType, ActivationType
from olmo.torch_util import get_global_rank
from olmo.util import resource_path

log = logging.getLogger(__name__)


class VisionBackboneType(StrEnum):
    openai = "openai"
    siglip = "siglip"
    dino = "dino"


class SdpaBackendType(StrEnum):
    """SDPA backend to use"""
    math = "math"
    efficient = "efficient"
    cudnn = "cudnn"
    flash = "flash"
    all = "all"


@dataclass
class VitConfig(BaseConfig):
    """Config for a vision transformer"""
    image_model_type: VisionBackboneType = VisionBackboneType.openai
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 24
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: ActivationType = ActivationType.gelu
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02

    float32_attention: bool = True

    attention_type: AttentionType = AttentionType.sdpa

    sdpa_backend: SdpaBackendType = SdpaBackendType.all

    activation_checkpointing: bool = True
    """Allow activation checkpointing for each layer"""

    init_path: Optional[str] = None
    """Path to initialize the ViT with"""

    resize_mode: str = "default"
    """How to resize images for this ViT"""

    pad_value: float = 0
    """Value to pad images with if the resize model pads image"""

    normalize: str = "openai"
    """How to normalize images for this ViT"""

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "truncate_pos_ids" in config:
            del config["truncate_pos_ids"]
        return config

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size

    def build(self, device):
        if self.image_model_type == VisionBackboneType.openai:
            return VisionTransformer(self)
        elif self.image_model_type == VisionBackboneType.siglip:
            return SiglipVisionTransformer(self)
        elif self.image_model_type == VisionBackboneType.dino:
            return DinoVisionTransformer(self)
        else:
            raise NotImplementedError(f"Unknown image model type: {self.image_model_type}")


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def vit_activation_checkpoint_function(cfg: VitConfig):
    preserve_rng_state = (
        (cfg.attention_dropout != 0.0) or (cfg.residual_dropout != 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class ViTMultiHeadDotProductAttention(nn.Module):
    """MDPA for the image ViT"""

    def __init__(self, config: VitConfig, use_bias: bool = True, input_dim=None, out_dim=None, device=None):
        super().__init__()
        self.config = config
        self.use_bias = use_bias

        self.embed_dim = config.image_emb_dim
        self.num_heads = config.image_num_heads
        self.head_dim = config.image_head_dim
        self.num_key_value_heads = config.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = config.initializer_range

        if input_dim is None:
            input_dim = self.embed_dim

        self.wq = nn.Linear(
            input_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=device,
            )
        self.wk = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
            )
        self.wv = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        if out_dim == -1:
            self.wo = None
        else:
            self.wo = nn.Linear(
                self.num_heads * self.head_dim,
                out_dim or self.embed_dim,
                bias=use_bias,
                device=device,
            )
        self.attention_dropout: Optional[nn.Dropout] = None
        if config.attention_dropout > 0:
            self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        sdpa_backend_list = []
        if config.sdpa_backend in [SdpaBackendType.all, SdpaBackendType.flash]:
            sdpa_backend_list.append(SDPBackend.FLASH_ATTENTION)
        if config.sdpa_backend in [SdpaBackendType.all, SdpaBackendType.cudnn]:
            sdpa_backend_list.append(SDPBackend.CUDNN_ATTENTION)
        if config.sdpa_backend in [SdpaBackendType.all, SdpaBackendType.efficient]:
            sdpa_backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
        if config.sdpa_backend in [SdpaBackendType.all, SdpaBackendType.math]:
            sdpa_backend_list.append(SDPBackend.MATH)
        self.sdpa_backend_list = sdpa_backend_list

    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        if self.wo is not None:
            nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            if self.wo is not None:
                nn.init.constant_(self.wo.bias, 0)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None, attn_mask=None) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        if self.config.attention_type == AttentionType.direct:
            assert attn_mask is None
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.config.attention_type == AttentionType.sdpa:
            if self.config.float32_attention:
                xv = xv.to(torch.float32)
            if inputs_kv is not None:
                with sdpa_kernel(self.sdpa_backend_list):
                    attn_output = F.scaled_dot_product_attention(
                        xq.transpose(1, 2).contiguous(),
                        xk.transpose(1, 2).contiguous(),
                        xv.transpose(1, 2).contiguous(),
                        attn_mask=attn_mask,
                        is_causal=False,
                        dropout_p=self.config.attention_dropout
                    ).transpose(1, 2)
            else:
                attn_output = F.scaled_dot_product_attention(
                    xq.transpose(1, 2).contiguous(),
                    xk.transpose(1, 2).contiguous(),
                    xv.transpose(1, 2).contiguous(),
                    attn_mask=attn_mask,
                    is_causal=False,
                    dropout_p=self.config.attention_dropout
                ).transpose(1, 2)
        else:
            raise NotImplementedError(self.config.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        if self.wo is not None:
            attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class ViTMLP(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config

        self.w1 = nn.Linear(
            config.image_emb_dim,
            config.image_mlp_dim,
            bias=True,
            device=device,
        )
        # Activation function.
        self.act = get_activation(config.image_mlp_activations)
        self.w2 = nn.Linear(
            config.image_mlp_dim,
            config.image_emb_dim,
            bias=True,
            device=device,
        )

    def reset_parameters(self):
        v_cfg = self.config
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / v_cfg.image_emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / v_cfg.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class BlockCollection(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        self._activation_checkpoint_fn: Optional[Callable] = None
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config, device) for _ in range(config.image_num_layers)
        ])

    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            if self._activation_checkpoint_fn:
                x = self._activation_checkpoint_fn(r, x)
            else:
                x = r(x)
            hidden_states.append(x)
        return hidden_states


class DinoBlockCollection(nn.Module):

    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self._activation_checkpoint_fn: Callable = vit_activation_checkpoint_function(self.config)

        self.resblocks = nn.ModuleList([
            DinoResidualAttentionBlock(config) for _ in range(config.image_num_layers)
        ])

    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(r, x)
            else:
                x = r(x)
            hidden_states.append(x)
        return hidden_states


class ResidualAttentionBlock(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        self.attention = ViTMultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )
        self.ffn_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class DinoResidualAttentionBlock(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        self.attention = ViTMultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )
        self.ffn_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )
        self.lambda1 = nn.Parameter(
            torch.ones(config.image_emb_dim, device=device),
        )
        self.lambda2 = nn.Parameter(
            torch.ones(config.image_emb_dim, device=device),
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        nn.init.ones_(self.lambda1)
        nn.init.ones_(self.lambda2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x)) * self.lambda1
        x = x + self.feed_forward(self.ffn_norm(x)) * self.lambda2
        return x


class VisionTransformer(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        # class embeddings and positional embeddings
        self.scale = config.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(config.image_emb_dim, device=device),
        )
        self.num_prefix_tokens: int = 1
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.image_emb_dim, device=device),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.image_emb_dim,
            bias=False,
            device=device,
            )

        self.pre_ln = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
            device=device,
        )

        self.transformer = BlockCollection(config)

    def reset_parameters(self):
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.pre_ln.reset_parameters()
        self.transformer.reset_parameters()

    def reset_with_pretrained_weights(self):
        if self.config.init_path:
            log.info(f"Loading ViT init params from {self.config.init_path}")
            is_sharded = hasattr(self.transformer.resblocks[0], "unshard")
            if not is_sharded or get_global_rank() == 0:
                parent, name = self.config.init_path.rsplit("/", 1)
                state_dict_path = resource_path(parent, name, cache_dir=os.environ.get("MOLMO_CACHE_DIR"))
                assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
                state_dict = torch.load(state_dict_path, map_location="cpu")
            else:
                state_dict = {}

            if is_sharded:
                key_errors = dist_cp_sd.set_model_state_dict(
                    model=self,
                    model_state_dict=state_dict,
                    options=dist_cp_sd.StateDictOptions(
                        full_state_dict=True, broadcast_from_rank0=True, strict=False)
                )
            else:
                key_errors = self.load_state_dict(state_dict, strict=False)

            assert len(key_errors.missing_keys) == 0
            # Missing keys are okay since we might have loaded fewer layers then in the checkpoint,
            # But sanity check the missing keys just in case
            for key in key_errors.unexpected_keys:
                assert key.startswith("transformer.resblocks.")
        else:
            self.reset_parameters()

    def apply_fsdp2(self, *args, **kwargs):
        for block in self.transformer.resblocks:
            fully_shard(block, *args, **kwargs)
        fully_shard(self, *args, **kwargs)

    def apply_activation_checkpointing(self):
        if self.config.activation_checkpointing:
            self.transformer._activation_checkpoint_fn = vit_activation_checkpoint_function(self.config)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        # positional embeddings
        self.scale = config.image_emb_dim ** -0.5
        self.num_prefix_tokens: int = 0 # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.image_emb_dim, device=device),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.image_emb_dim,
            bias=True,
            device=device,
        )

        self.transformer = BlockCollection(config, device)

    def apply_activation_checkpointing(self):
        fn = vit_activation_checkpoint_function(self.config)
        if self.config.activation_checkpointing:
            self.transformer.resblocks = nn.ModuleList([
                checkpoint_wrapper(x, checkpoint_fn=fn) for x in self.transformer.resblocks])

    def reset_with_pretrained_weights(self):
        # FIXME just repeating the code in `VisionTransformer`
        if self.config.init_path:
            t0 = time.perf_counter()
            log.info(f"Loading ViT parameters from {self.config.init_path}")
            is_sharded = hasattr(self.transformer.resblocks[0], "unshard")

            if self.config.init_path.startswith("model:"):
                path = self.config.init_path[len("model:"):].rstrip("/") + "/model_and_optim"
                import torch.distributed.checkpoint as dist_cp
                from olmo.train.remote_filesystem import RemoteFileSystemReader
                assert is_sharded
                sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=False, strict=False)
                state_dict = dist_cp_sd.get_model_state_dict(self, options=sd_options)
                state_dict = {f"model.vision_backbone.image_vit.{k}": v for k, v in state_dict.items()}
                dist_cp.load(
                    state_dict,
                    checkpoint_id=path,
                    storage_reader=RemoteFileSystemReader(path),
                )
                state_dict = {k[len("model.vision_backbone.image_vit."):]: v for k, v in state_dict.items()}
                key_errors = dist_cp_sd.set_model_state_dict(
                    model=self,
                    model_state_dict=state_dict,
                    options=dist_cp_sd.StateDictOptions(strict=False)
                )
                log.info(f"Done in {time.perf_counter()-t0:0.1} seconds")
                return

            if not is_sharded or get_global_rank() == 0:
                parent, name = self.config.init_path.rsplit("/", 1)
                state_dict_path = resource_path(parent, name, cache_dir=os.environ.get("MOLMO_CACHE_DIR"))
                assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
                state_dict = torch.load(state_dict_path, map_location="cpu")
            else:
                state_dict = {}

            if is_sharded:
                key_errors = dist_cp_sd.set_model_state_dict(
                    model=self,
                    model_state_dict=state_dict,
                    options=dist_cp_sd.StateDictOptions(
                        full_state_dict=True, broadcast_from_rank0=True, strict=False)
                )
            else:
                key_errors = self.load_state_dict(state_dict, strict=False)

            assert len(key_errors.missing_keys) == 0
            # Missing keys are okay since we might have loaded fewer layers then in the checkpoint,
            # But sanity check the missing keys just in case
            for key in key_errors.unexpected_keys:
                assert key.startswith("transformer.resblocks.")
            log.info(f"Done in {time.perf_counter()-t0:0.1} seconds")
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        nn.init.zeros_(self.patch_embedding.bias)
        self.transformer.reset_parameters()

    def apply_fsdp2(self, *args, **kwargs):
        for block in self.transformer.resblocks:
            fully_shard(block, *args, **kwargs)
        fully_shard(self, *args, **kwargs)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.transformer(x)
        return hidden_states


class DinoVisionTransformer(nn.Module):

    def __init__(self, config: VitConfig, device=None):
        super().__init__()
        self.config = config
        # class embeddings and positional embeddings
        self.scale = config.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(config.image_emb_dim, device=device),
        )
        self.num_prefix_tokens: int = 1
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.image_emb_dim, device=device),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.image_emb_dim,
            bias=True,
            device=device,
        )

        self.transformer = DinoBlockCollection(config)

    def apply_activation_checkpointing(self):
        if self.config.activation_checkpointing:
            self.transformer._activation_checkpoint_fn = vit_activation_checkpoint_function(self.config)

    def reset_parameters(self):
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.transformer.reset_parameters()

    def full_shard(self, *args, **kwargs):
        for block in self.transformer.resblocks:
            fully_shard(block, *args, **kwargs)
        fully_shard(self, *args, **kwargs)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.transformer(x)
        return hidden_states
