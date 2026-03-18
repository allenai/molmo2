import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Callable, Any, List, Tuple

import numpy as np
import torch
from torch import nn

from torch.nn import functional as F
from transformers import LogitsProcessorList, LogitsProcessor, AutoProcessor, ViTConfig
from transformers.image_utils import PILImageResampling

from transformers.models.auto import AutoModelForImageTextToText
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_masks_for_generate
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    FlashAttentionKwargs,
    flash_attn_supports_top_left_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    can_return_tuple,
    logging,
)

from .configuration_molmo2 import Molmo2VitConfig, Molmo2TextConfig, Molmo2AdapterConfig
from .configuration_molmo_point import MolmoPointConfig, MolmoPointAdapterConfig
from .image_processing_molmo2 import Molmo2ImagesKwargs, image_to_patches_and_grids
from .modeling_molmo2 import ImageProjectorMLP, Molmo2VisionTransformer, Molmo2RMSNorm, \
    Molmo2RotaryEmbedding, Molmo2PostNormDecoderLayer, Molmo2DecoderLayer, Molmo2Attention, \
    Molmo2Embedding

# FIXME remove
processor = None
def decode(ids):
    global processor
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            "/weka/oe-training-default/mm-olmo/released-models-molmo2-point-0326/MolmoPoint-8B/hf-step2000", trust_remote_code=True,
            padding_side="left")
    return processor.post_process_image_text_to_text(ids.view(1), skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]


logger = logging.get_logger(__name__)
NO_POINTS_LABEL = 1000000


EXTRACT_POINT_TRIPLE = re.compile(f"<POINT_(\d+)> ?<POINT_(\d+)> ?<POINT_(\d+)> ?([0-9]+)" )


def get_subpatch_ids(output_text, pooling, no_more_points_class):
    n_patches, n_subpatches = pooling.shape[-2:]
    if no_more_points_class:
        n_patches += 1
    for match in EXTRACT_POINT_TRIPLE.finditer(output_text):
        patch_id, subpatch_num = int(match.group(1)), int(match.group(2))
        subpatch_id = subpatch_num - n_patches
        location_num = int(match.group(3))
        location_id = location_num - n_patches - n_subpatches
        example_id = int(match.group(4))
        vit_patch_id = pooling[patch_id, subpatch_id]
        yield vit_patch_id, location_id, example_id


@dataclass
class ImageCache:
    """Extra stuff we need to cache when doing autoregressive generation with pointing"""

    patch_k: torch.FloatTensor
    """K values of the image tokens"""

    patch_k_mask: torch.BoolTensor
    """Mask over image tokens that can be selected"""

    subpatch_k: torch.FloatTensor
    """K values of the ViT patches before pooling"""

    token_pooling: torch.LongTensor
    """token pooling array mapping image_patch_id -> ViT patches pooled for that patch"""

    vit_features: torch.FloatTensor
    """Features before pooling, used for building input embeddings"""

    image_pos_ids: Optional[torch.LongTensor] = None
    """Position ids of the image tokens if need for rotary embeddings"""

    image_features0: Optional[torch.FloatTensor] = None
    """"Image features, might be needed to embed new patch prediction tokens"""

    flat_image_tokens_to_flat_image_features: Optional[torch.LongTensor] = None
    """Cached for indexing uses"""


@dataclass
class MolmoPointCausalLMOutputWithPast(ModelOutput):
    """
    Base class for MolmoPoint causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    image_data: Optional[ImageCache] = None
    patch_logits: Optional[torch.FloatTensor] = None
    subpatch_logits: Optional[torch.FloatTensor] = None
    location_logits: Optional[torch.FloatTensor] = None
    last_predicted_patch_id: Optional[torch.LongTensor] = None


@dataclass
class MolmoPointModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for Molmo2 outputs, with hidden states and attentions.

    Args:
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_num_patches, hidden_size)`.
            image_hidden_states of the model produced by the vision backbone
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    image_data: Optional[ImageCache] = None
    patch_logits: Optional[torch.FloatTensor] = None
    subpatch_logits: Optional[torch.FloatTensor] = None
    location_logits: Optional[torch.FloatTensor] = None
    input_ids: Optional[torch.LongTensor] = None
    last_predicted_patch_id: Optional[torch.LongTensor] = None


class MolmoPointPatchRope(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(
        self,
        theta: float,
        dim: int,
        device: Union[str, torch.device] = None,
    ):
        super().__init__()
        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, hs = x.size()
        x = x.view(B, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    @torch.no_grad()
    def forward(self, x, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq.float().to(x.device)
        position_ids_expanded = position_ids.float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            x = x.float()
            freqs = position_ids_expanded[:, None] * inv_freq_expanded[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            out = ((x * cos) + (self.rotate_half(x) * sin))

        return out.to(dtype=x.dtype)


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: Optional[int] = None,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        device: Union[str, torch.device] = None,
        attn_implementation: str = "eager",
        out_layer: bool=True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_implementation = attn_implementation
        self.is_causal = False

        input_dim = input_dim or hidden_size

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
        if out_layer:
            self.wo = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                )
        else:
            self.wo = None
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = nn.Dropout(residual_dropout)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

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

        if self.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        dropout_p = 0.0 if not self.training else self.attention_dropout

        if self.attn_implementation == "eager":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            attn_weights = F.dropout(
                attn_weights,
                p=dropout_p,
                training=self.training
            )
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.attn_implementation == "sdpa":
            if not torch.is_autocast_enabled():
                xv = xv.to(torch.float)

            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=dropout_p,
            ).transpose(1, 2)

        elif self.attn_implementation == "flash_attention_2":
            if xq.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                else:
                    target_dtype = self.wq.weight.dtype
            attn_output = _flash_attention_forward(
                xq,
                xk,
                xv,
                attention_mask=attn_mask,
                query_length=inputs_q.shape[1],
                is_causal=False,
                dropout=dropout_p,
                softmax_scale=xq.shape[-1] ** -0.5,
                use_top_left_mask=flash_attn_supports_top_left_mask(),
                target_dtype=target_dtype,
                implementation=self.attn_implementation,
            )
        else:
            raise ValueError(f"Attention implementation {self.attn_implementation} not supported")

        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        if self.wo is not None:
            attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class PointPredictor(nn.Module):
    """Point predictor logic"""
    # We separate this out so accelerate will co-locate all these parameters on the same device

    def __init__(self, config):
        super().__init__()
        self.config = config
        llm_dim = config.text_config.hidden_size
        patch_embed_dim = config.patch_embed_dim
        vit_dim = self.config.vit_config.hidden_size * len(self.config.adapter_config.vit_layers)
        if self.config.layer_norm_x:
            self.x_norm = Molmo2RMSNorm(llm_dim, eps=self.config.text_config.layer_norm_eps)
        else:
            self.x_norm = None
        if self.config.token_prediction_rotary == "none":
            self.patch_rotary = None
        else:
            theta = self.config.token_prediction_rotary_theta or self.config.llm.rope_theta
            if self.config.token_prediction_rotary == "one_d":
                self.patch_rotary = MolmoPointPatchRope(theta, self.config.patch_embed_dim)
            else:
                raise NotImplementedError()
        self.patch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.patch_k = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_k = nn.Linear(vit_dim, patch_embed_dim)
        self.add_no_point_class_embed = MolmoPointPadWithLearnedVector(patch_embed_dim)
        if self.config.patch_location == "3x3":
            self.subpatch_loc_k = nn.Linear(llm_dim, 9)
        elif self.config.patch_location is None:
            self.subpatch_loc_k = None
        else:
            raise NotImplementedError(f"Patch location {self.config.patch_location} not implemented")

    def forward(
        self,
        x,
        token_pooling,
        is_image_token,
        is_patch,
        is_subpatch,
        is_indexable_image_token,
        vit_features,
        vit_features_mask,
        image_features_mask,
        input_patch_ids,
        last_predicted_patch_id,
        image_data: ImageCache
    ):
        dim = self.config.text_config.hidden_size
        batch_size = x.shape[0]
        if self.x_norm is not None:
            x_norm = self.x_norm(x)
        elif self.config.norm_x:
            x_norm = x / math.sqrt(dim)
        else:
            x_norm = x

        # Build the keys, or get them from the cache
        if image_data is not None:
            patch_k, subpatch_k = image_data.patch_k, image_data.subpatch_k
            patch_k_mask = image_data.patch_k_mask
            token_pooling = image_data.token_pooling
            vit_features_mask = token_pooling >= 0
            image_pos_ids = image_data.image_pos_ids
        else:
            # Build patch keys, this takes a bit of indexing trickery since we want the keys in
            # shape [batch, n_image_tokens] not [batch, sequence_length]
            n_image_tokens = token_pooling.shape[1]
            patch_k_flat = self.patch_k(x_norm.view(-1, dim)[is_image_token.view(-1)])
            if self.patch_rotary is not None:
                image_token_indices = torch.cumsum(is_indexable_image_token, dim=-1) - 1
                image_pos_ids_flat = image_token_indices.view(-1)[is_image_token.view(-1)]
                patch_k_flat = self.patch_rotary(patch_k_flat, image_pos_ids_flat)

                # Computed for use with the query vectors
                image_pos_ids = torch.zeros([batch_size, n_image_tokens], dtype=torch.long,
                                            device=image_pos_ids_flat.device)
                image_pos_ids.view(-1)[image_features_mask.view(-1)] = image_pos_ids_flat
            else:
                image_pos_ids = None

            patch_k = torch.zeros([batch_size, n_image_tokens, patch_k_flat.shape[-1]],
                                  dtype=x.dtype, device=x.device)
            patch_k.view(-1, patch_k_flat.shape[-1])[image_features_mask.flatten()] = patch_k_flat.to(dtype=x.dtype)

            patch_k_mask = image_features_mask.clone()
            patch_k_mask.view(-1)[image_features_mask.view(-1)] = (
                is_indexable_image_token.view(-1)[is_image_token.view(-1)])

            if self.config.no_more_points_class:
                patch_k = self.add_no_point_class_embed(patch_k)
                patch_k_mask = F.pad(patch_k_mask, (0, 1), value=True)

            subpatch_k = self.subpatch_k(vit_features)

        patch_logits, subpatch_logits, location_logits = None, None, None
        if image_data is not None:
            # Predict patch locations, only done after pre-filling
            batch_idx = torch.arange(batch_size, device=x_norm.device)
            image_q = self.patch_q(x_norm)
            if self.patch_rotary is not None and last_predicted_patch_id is not None:
                rotate_by = image_pos_ids[batch_idx, last_predicted_patch_id]
                rotate_by = torch.where(last_predicted_patch_id >= 0, rotate_by, 0)
                rotate_by = rotate_by.squeeze(-1)
                image_q = self.patch_rotary(
                    image_q.view(-1, image_q.shape[-1]),
                    torch.clamp(rotate_by, min=0),
                ).reshape(batch_size, -1, image_q.shape[-1])

            dots = torch.matmul(image_q, patch_k.transpose(1, 2))  # [batch, 1, num_images]
            if self.config.norm_logits:
                dots = dots / math.sqrt(dots.shape[-1])

            valid = patch_k_mask[:, None, :]
            patch_logits = torch.where(valid, dots, -100000000)

            if torch.any(is_patch):
                if x_norm.shape[1] != 1:
                    raise NotImplementedError()
                subpatch_point_q = self.subpatch_q(x_norm.squeeze(1))
                subpatch_k = subpatch_k[batch_idx, input_patch_ids.squeeze(1)]
                subpatch_logits = torch.einsum("pd,pcd->pc", subpatch_point_q, subpatch_k)
                if self.config.norm_logits:
                    subpatch_logits = subpatch_logits / math.sqrt(patch_k.shape[-1])
                subpatch_mask = vit_features_mask[batch_idx, input_patch_ids.squeeze(1)]
                subpatch_logits = torch.where(subpatch_mask, subpatch_logits, -100000)
                subpatch_logits = subpatch_logits[:, None, :]

            if torch.any(is_subpatch):
                location_logits = self.subpatch_loc_k(x)

        if image_data is None:
            image_data = ImageCache(
                patch_k=patch_k,
                subpatch_k=subpatch_k,
                vit_features=vit_features,
                patch_k_mask=patch_k_mask,
                token_pooling=token_pooling,
                image_pos_ids=image_pos_ids,
            )
        return patch_logits, subpatch_logits, location_logits, image_data


class MolmoPointPreTrainedModel(PreTrainedModel):
    config: MolmoPointConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Molmo2DecoderLayer",
        "Molmo2PostNormDecoderLayer",
        "Molmo2VisionBlock",
        "ViTMultiHeadDotProductAttention",
        "PointPredictor"
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Molmo2Embedding):
            module.embedding.data.normal_(mean=0.0, std=std)
            module.new_embedding.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Molmo2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class GeneratedTokenBounds:
    """Describes what tokens id ranges are patch/subpatch/location tokens"""

    def __init__(self, vocab_size, n_patches, n_subpatches, n_locations, no_more_points_class):
        self.n_locations = n_locations
        self.n_patches = n_patches
        self.n_subpatches = n_subpatches
        self.vocab_size = vocab_size

        if no_more_points_class:
            self.no_more_points_token_id = vocab_size + n_patches
        else:
            self.no_more_points_token_id = -1
        self.patch_start = vocab_size
        self.patch_end_without_no_more_points = vocab_size + n_patches
        self.patch_end = vocab_size + n_patches + int(no_more_points_class)
        self.subpatch_start = self.patch_end
        self.subpatch_end = self.subpatch_start + n_subpatches
        self.location_start = self.subpatch_end
        self.location_end = self.subpatch_end + n_locations


class MolmoPointLogitProcessor(LogitsProcessor):
    """Force point-special tokens to be generated in a valid order"""

    def __init__(self, bounds: GeneratedTokenBounds,
                 prevent_repeats, force_patch_sorted, force_subpatch_sorted):
        self.bounds = bounds
        self.prevent_repeats = prevent_repeats
        self.force_patch_sorted = force_patch_sorted
        self.force_subpatch_sorted = force_subpatch_sorted

    def __call__(self, input_ids, scores):
        b = self.bounds
        is_complete_patch = (b.patch_start <= input_ids) & (input_ids < b.patch_end)
        is_complete_subpatch = (b.subpatch_start <= input_ids) & (input_ids < b.subpatch_end)

        if b.n_locations:
            is_complete_patch[:, -2:] = False
            is_complete_subpatch[:, -2:] = False
        else:
            is_complete_patch[:, -1] = False
            is_complete_subpatch[:, -1] = False

        for batch in range(len(input_ids)):
            batch_input_ids = input_ids[batch]
            last_token = batch_input_ids[-1]

            batch_is_patch_token = is_complete_patch[batch]
            last_predicted_patch_token = batch_input_ids[is_complete_patch[batch]]
            if len(last_predicted_patch_token):
                last_predicted_patch_token = last_predicted_patch_token[-1]
            else:
                last_predicted_patch_token = None

            last_predicted_subpatch_token = batch_input_ids[is_complete_subpatch[batch]]
            if len(last_predicted_subpatch_token):
                last_predicted_subpatch_token = last_predicted_subpatch_token[-1]
            else:
                last_predicted_subpatch_token = None

            no_more_points = torch.any(batch_input_ids == b.no_more_points_token_id)

            if no_more_points:
                # Cannot generate any kind of point
                scores[batch, b.patch_start:b.location_end] = -float("inf")
            elif last_token < b.patch_start or last_token >= b.subpatch_end:
                # Cannot generate subpatch/location, but might generate a patch
                scores[batch, b.subpatch_start:b.location_end] = -float("inf")

                if self.force_patch_sorted and last_predicted_patch_token is not None:
                    # Cannot generate patches that occurs before the previously predicted patch
                    scores[batch, b.patch_start:last_predicted_patch_token] = -float("inf")

                if (
                    self.prevent_repeats and
                    self.force_subpatch_sorted and
                    last_predicted_subpatch_token is not None and
                    last_predicted_subpatch_token == (b.subpatch_end-1)
                ):
                    # Generating `last_predicted_patch_token` would force us to generate a repeat
                    # since the only subpatch we can predict while keeping sorted order
                    # will repeat the previous point
                    scores[batch, last_predicted_patch_token] = -float("inf")

            elif b.patch_start <= last_token < b.patch_end:
                # Last token was a patch token, must select a subpatch next
                scores[batch, :b.subpatch_start] = -float("inf")
                scores[batch, b.subpatch_end:] = -float("inf")
                if (
                    self.force_subpatch_sorted and
                    last_predicted_patch_token == last_token
                ):
                    assert last_predicted_subpatch_token is not None
                    if self.prevent_repeats:
                        assert last_predicted_subpatch_token != b.subpatch_end-1
                        scores[batch, b.subpatch_start:last_predicted_subpatch_token+1] = -float("inf")
                    else:
                        scores[batch, b.subpatch_start:last_predicted_subpatch_token] = -float("inf")

            elif b.n_locations and b.subpatch_start <= last_token < b.subpatch_end:
                # Last token was a subpatch token, must select a location next
                scores[batch, :b.location_start] = -float("inf")
                scores[batch, b.location_end:] = -float("inf")
            else:
                raise RuntimeError("Unreachable")
        return scores


@dataclass
class Molmo2TextBaseOutput(BaseModelOutputWithPast):
    pre_ln_hidden_state: Optional[torch.FloatTensor] = None


class MolmoPointTextModel(PreTrainedModel):
    config: Molmo2TextConfig
    _no_split_modules = ["Molmo2DecoderLayer", "Molmo2PostNormDecoderLayer"]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def __init__(self, config: Molmo2TextConfig):
        super().__init__(config)
        if config.additional_vocab_size is not None:
            self.wte = Molmo2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        decoder_layer = Molmo2PostNormDecoderLayer if config.norm_after else Molmo2DecoderLayer
        self.blocks = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.ln_f = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.rope_scaling_layers is not None:
            self.rotary_embs = nn.ModuleDict(
                {
                    "default": Molmo2RotaryEmbedding(config, rope_type="default"),
                    "scaling": Molmo2RotaryEmbedding(config),
                }
            )
        else:
            self.rotary_emb = Molmo2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.wte = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_pre_ln_state: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Molmo2TextBaseOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            # Create the mask
            causal_mask_mapping = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": self.rotary_embs["default"](hidden_states, position_ids),
                "scaling": self.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_block in enumerate(self.blocks[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in self.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                hidden_states,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings_i,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        pre_ln_state = hidden_states
        hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return Molmo2TextBaseOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            pre_ln_hidden_state=pre_ln_state,
            hidden_states=hidden_states,
            attentions=all_self_attns,
        )

# Adapted from transformers.models.gemma3.modeling_gemma3
def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor] = None,
) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
        token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)

        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & is_image_block

    return inner_mask


class MolmoPointPadWithLearnedVector(nn.Module):
    """Module that pads vector

    Used to add in the no-more-point key value
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.vector = nn.Parameter(torch.zeros([dim]))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vector = torch.tile(self.vector[None, :], [x.shape[0], 1])
        return torch.concatenate([x, vector[:, None, :]], dim=1)


class AddPosEmbed(nn.Module):

    def __init__(self, in_features: int, n_pos: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([n_pos, in_features]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias[None, :input.shape[-2], :]


class MolmoPointAdapter(nn.Module):
    def __init__(self, config: MolmoPointAdapterConfig, vit_config: Molmo2VitConfig):
        super().__init__()
        self.config = config
        self.n_vit_layers = len(config.vit_layers)
        pool_dim = vit_config.hidden_size * self.n_vit_layers
        self.norm = None
        self.image_projector = ImageProjectorMLP(
            config.hidden_size,
            config.intermediate_size,
            config.text_hidden_size,
            config.hidden_act,
        )
        self.act = ACT2FN[config.hidden_act]
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            input_dim=pool_dim,
            float32_attention=config.float32_attention,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            attn_implementation=config._attn_implementation,
            out_layer=config.attention_pooling_out_layer
        )
        if self.config.positional_embeddings:
            self.positional_embeddings = AddPosEmbed(pool_dim, self.config.positional_embeddings)
        else:
            self.positional_embeddings = None

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


def extract_image_points(output_text, pooling, mappings, no_more_points_class, location, image_sizes):
    """Extract points from MolmoPoint image output text

    return points: [n_points, 4] array of (object_id, image_num, x, y) points
    """
    if len(mappings) != len(image_sizes):
        raise ValueError("Mapping and image sizes must have the same length")
    extracted_points = []
    for vit_patch_id, location_id, example_id in get_subpatch_ids(output_text, pooling, no_more_points_class):
        for image_ix, (mapping, (w, h)) in enumerate(zip(mappings, image_sizes)):
            patch_coords = np.argwhere(mapping == int(vit_patch_id))
            if len(patch_coords) == 1:
                p_y, p_x = patch_coords[0]
                if location_id is not None:
                    loc_x = location_id // 3
                    loc_y = location_id % 3
                    p_x += (loc_x+0.5)*0.33
                    p_y += (loc_y+0.5)*0.33
                else:
                    p_x += 0.5
                    p_y += 0.5
                extracted_points.append([
                    example_id,
                    image_ix,
                    (p_x / mapping.shape[1]) * w,
                    (p_y / mapping.shape[0]) * h,
                    ])
                break
        else:
            logger.error("Invalid patch id encountered")
    return extracted_points


def extract_video_points(output_text, pooling, mapping, timestamps, no_more_points_class,
                         location, video_size):
    """
    Extract points from MolmoPoint video output text

    return points: [n_points, 4] array of (object_id, timestamp, x, y) points
    """
    extracted_points = []
    for vit_patch_id, location_id, example_id in get_subpatch_ids(output_text, pooling, no_more_points_class):
        patch_coords = np.argwhere(mapping == int(vit_patch_id))
        if len(patch_coords) == 1:
            frame_ix, p_y, p_x = patch_coords[0]
            if location_id is not None:
                loc_x = location_id // 3
                loc_y = location_id % 3
                p_x += (loc_x+0.5)*0.33
                p_y += (loc_y+0.5)*0.33
            else:
                p_x += 0.5
                p_y += 0.5
            ts = timestamps[frame_ix]
            extracted_points.append([
                example_id,
                ts,
                (p_x / mapping.shape[2]) * video_size[0],
                (p_y / mapping.shape[1]) * video_size[1]
            ])
        else:
            logger.error("Invalid patch id encountered")
    return extracted_points


class MolmoPointModel(MolmoPointPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MolmoPointConfig

    def __init__(self, config: MolmoPointConfig):
        super().__init__(config)
        self.transformer: MolmoPointTextModel = MolmoPointTextModel(config.text_config)
        self.patch_token_id = self.config.patch_token_id
        self.subpatch_token_id = self.config.subpatch_token_id
        self.location_token_id = self.config.location_token_id

        vit_config = config.vit_config
        adapter_config = config.adapter_config
        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)

        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            new_vit_config = deepcopy(vit_config)
            new_vit_config.num_hidden_layers = last_layer_needed
            self.vit = Molmo2VisionTransformer(new_vit_config)
        else:
            self.vit = Molmo2VisionTransformer(vit_config)

        self.connector = MolmoPointAdapter(adapter_config, vit_config)
        if self.config.embed_selected_vit_patch == "linear":
            llm_dim = config.text_config.hidden_size
            vit_dim = self.config.vit_config.hidden_size * len(self.config.adapter_config.vit_layers)
            self.build_vit_embedding = nn.Linear(vit_dim, llm_dim, bias=True)
        else:
            raise NotImplementedError(f"Embedding {self.config.embed_selected_vit_patch} not implemented")
        self.point_predictor = PointPredictor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def build_token_bounds(self, token_pooling):
        n_patches, n_subpatches = token_pooling.shape[-2:]
        return GeneratedTokenBounds(
            vocab_size=self.config.vocab_size + self.config.text_config.additional_vocab_size,
            n_patches=n_patches,
            n_subpatches=n_subpatches,
            n_locations=9 if self.config.patch_location else 0,
            no_more_points_class=self.config.no_more_points_class,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.transformer.wte = value

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def build_batched_images(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Count the number of images in each example
        raw_counts = (input_ids == self.config.image_end_token_id).sum(1)  # [N]
        # Each image is represented by global view and high-res view
        # so we divide by 2 to get the number of images
        counts = raw_counts // 2
        N = counts.size(0)
        device = input_ids.device

        # Total number of images in the batch
        num_images = int(counts.sum().item())

        # Sanity check
        assert image_grids.size(0) == num_images, \
            f"Expected {num_images} image grids, but got {image_grids.size(0)}"
        assert image_num_crops.size(0) == num_images, \
            f"Expected {num_images} image num crops, but got {image_num_crops.size(0)}"

        # 1-1) Compute per-image pooled patch count from image grids
        with torch.no_grad():
            first_prod = image_grids[:, :2].prod(dim=1)    # [num_images]
            second_prod = image_grids[:, 2:].prod(dim=1)   # [num_images]
            num_pooled_patches_per_image = (first_prod + second_prod).to(image_num_crops.dtype)  # [num_images]

        # pixel_values: [n_crops, n_patches, pixels_per_patch]
        n_crops, n_patches, pixels_per_patch = pixel_values.shape

        # 2) Map each image index → example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_image = torch.arange(N, device=device).repeat_interleave(counts)  # [num_images]
        assert example_ids_for_image.numel() == num_images

        # 2-1) Compute crops_per_example by summing per-image crop counts
        crops_per_example = torch.zeros(
            N, dtype=image_num_crops.dtype, device=image_num_crops.device
        )
        crops_per_example.index_add_(0, example_ids_for_image, image_num_crops)  # [N]

        # 2-2) Per-image number of patches = (crops per image) * n_patches
        patches_per_image = image_num_crops * n_patches  # [num_images]

        # 2-3) Compute per-example per-image patch offsets
        counts_list = counts.tolist()
        index_offset_per_example_list = []
        offset_img = 0
        for c in counts_list:
            per_img_patches = patches_per_image[offset_img:offset_img + c]  # [c]
            # Offsets: [0, img0_total_patches, img0+img1_total_patches, ...]
            index_offset = [0] + per_img_patches.cumsum(0).tolist()[:-1]
            index_offset_per_example_list.append(index_offset)
            offset_img += c

        # 2-4) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N, dtype=num_pooled_patches_per_image.dtype, device=num_pooled_patches_per_image.device
        )
        num_pooled_patches_per_example.index_add_(
            0, example_ids_for_image, num_pooled_patches_per_image
        )

        # Sanity checks
        total_crops = int(crops_per_example.sum().item())
        assert total_crops == n_crops, \
            f"Expected {total_crops} crops, but got {n_crops}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == image_token_pooling.size(0), \
            f"Expected {total_num_pooled_patches} pooled patches, but got {image_token_pooling.size(0)}"

        # 3) Build images tensor filled with -1
        M = int(crops_per_example.max().item())
        images = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        # 4) Fill images with per-example slices from pixel_values
        offset_crop = 0
        for i in range(N):
            num = int(crops_per_example[i].item())
            cur = pixel_values[offset_crop:offset_crop + num]  # [num, n_patches, pixels_per_patch]
            images[i, :num] = cur
            offset_crop += num

        # Sanity check
        assert offset_crop == n_crops

        # 5) Build new_token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = image_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=image_token_pooling.dtype,
            device=image_token_pooling.device,
        )

        # 6) Fill token_pooling with per-example slices, adding per-image patch offsets
        patch_offset = 0
        img_offset = 0

        for i, c in enumerate(counts_list):
            num_patches = int(num_pooled_patches_per_example[i].item())

            # Subsequence of pooled tokens belonging to this example
            cur = image_token_pooling[patch_offset:patch_offset + num_patches].clone()  # [num_patches, dim]

            index_offset_per_example = index_offset_per_example_list[i]  # length = c
            per_img_pooled = num_pooled_patches_per_image[img_offset:img_offset + c]   # [c]

            assert len(index_offset_per_example) == per_img_pooled.numel()

            # Apply per-image offsets to the (ragged) subsequence
            offset = 0
            for j in range(c):
                index_offset = int(index_offset_per_example[j])
                n = int(per_img_pooled[j].item())
                cur_slice = cur[offset:offset + n]

                # Apply offset across all columns
                cur[offset:offset + n] = torch.where(
                    cur_slice >= 0,
                    cur_slice + index_offset,
                    cur_slice,
                    )
                offset += n

            new_token_pooling[i, :num_patches] = cur

            patch_offset += num_patches
            img_offset += c

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches
        assert img_offset == num_images

        return images, new_token_pooling

    def build_batched_videos(
        self,
        input_ids: torch.LongTensor,
        pixel_values_videos: torch.Tensor,
        video_token_pooling: torch.Tensor,
        video_grids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1) Count the number of videos in each example
        if self.config.use_frame_special_tokens:
            end_token_id = self.config.frame_end_token_id
        else:
            end_token_id = self.config.image_end_token_id
        counts = (input_ids == end_token_id).any(dim=1).long()  # [N]
        N = counts.size(0)
        device = input_ids.device

        # Total number of videos in the batch
        num_videos = int(counts.sum().item())

        # Sanity check
        assert video_grids.size(0) == num_videos, \
            f"Expected {num_videos} videos, but got {video_grids.size(0)}"

        video_num_frames = video_grids[:, 0]  # [num_videos]
        num_pooled_patches_per_video = video_grids.prod(dim=1)  # [num_videos]

        # pixel_values_videos: [n_frames, n_patches, pixels_per_patch]
        n_frames, n_patches, pixels_per_patch = pixel_values_videos.shape

        # 2) Map each video index -> example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_video = torch.arange(N, device=device).repeat_interleave(counts)  # [num_videos]
        assert example_ids_for_video.numel() == num_videos

        # 2-1) Compute frames_per_example by summing per-video frame counts
        frames_per_example = torch.zeros(
            N, dtype=video_num_frames.dtype, device=device,
        )
        frames_per_example.index_add_(0, example_ids_for_video, video_num_frames)  # [N]

        # 2-2) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N, dtype=num_pooled_patches_per_video.dtype, device=num_pooled_patches_per_video.device,
        )
        num_pooled_patches_per_example.index_add_(
            0, example_ids_for_video, num_pooled_patches_per_video,
        )

        # Sanity checks
        total_frames = int(frames_per_example.sum().item())
        assert total_frames == n_frames, \
            f"Expected {total_frames} frames, but got {n_frames}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == video_token_pooling.size(0), \
            f"Expected {total_num_pooled_patches} pooled patches, but got {video_token_pooling.size(0)}"

        # 3) Build videos tensor filled with -1
        M = int(frames_per_example.max().item())
        videos = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values_videos.dtype,
            device=device,
        )

        # 4) Fill videos with per-examples slices from pixel_values_videos
        offset_frame = 0
        for i in range(N):
            num = int(frames_per_example[i].item())
            cur = pixel_values_videos[offset_frame:offset_frame + num]  # [num, n_patches, pixels_per_patch]
            videos[i, :num] = cur
            offset_frame += num

        # Sanity check
        assert offset_frame == n_frames

        # 5) Build new token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = video_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=video_token_pooling.dtype,
            device=video_token_pooling.device,
        )

        # 6) Fill new token_pooling with per-examples slices from video_token_pooling
        patch_offset = 0
        for i in range(N):
            num_patches = int(num_pooled_patches_per_example[i].item())
            cur = video_token_pooling[patch_offset:patch_offset + num_patches]  # [num_patches, dim]
            new_token_pooling[i, :num_patches] = cur
            patch_offset += num_patches

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches

        return videos, new_token_pooling

    def merge_visual_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")
        elif pixel_values is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_images(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )
        elif pixel_values_videos is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_videos(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )
        else:
            images, token_pooling = None, None
        return images, token_pooling

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,

        image_data: Optional[ImageCache] = None,
        last_predicted_patch_id: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MolmoPointModelOutputWithPast]:
        """
        last_point_patch_id: The patch id the last generated point pointed to
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        images, token_pooling = self.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )
        if inputs_embeds is not None:
            raise NotImplementedError("Custom inputs_embeds is not implemented yet")

        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        if image_data is not None:
            # Figure out where the patch/subpatch/location are and their values, and then convert
            # the input_ids back into their original special token values
            can_point = True
            bounds = self.build_token_bounds(image_data.token_pooling)
            expanded_inputs = input_ids
            is_patch = (input_ids >= bounds.patch_start) & (input_ids < bounds.patch_end_without_no_more_points)
            is_no_more_points = (input_ids == bounds.no_more_points_token_id)
            is_subpatch = (input_ids >= bounds.subpatch_start) & (input_ids < bounds.subpatch_end)
            is_location = (input_ids >= bounds.location_start) & (input_ids < bounds.location_end)
            input_patch_ids = torch.where(is_patch, input_ids - bounds.patch_start, -1)
            input_subpatch_ids = torch.where(is_subpatch, input_ids - bounds.subpatch_start, -1)
            input_ids = torch.where(is_patch | is_no_more_points, self.patch_token_id, input_ids)
            input_ids = torch.where(is_subpatch, self.subpatch_token_id, input_ids)
            input_ids = torch.where(is_location, self.location_token_id, input_ids)
        else:
            # No patch prediction during pre-filling
            input_subpatch_ids = None
            input_patch_ids = None
            is_patch = None
            is_subpatch = None
            can_point = False

        device = input_ids.device
        x = self.transformer.wte(input_ids).to(device=device)
        batch_size, _, dim = x.shape
        batch_idx = torch.arange(batch_size, device=device)

        vit_features_flat: Optional[torch.FloatTensor] = None
        if images is not None:
            is_indexable_image_token = input_ids == self.config.image_patch_id
            is_non_indexable_image_token = input_ids == self.config.image_non_indexable_patch_id
            is_image_token = is_indexable_image_token | is_non_indexable_image_token

            images = images.to(device=self.device, dtype=self.dtype)
            B, T, N, D = images.shape
            images = images.view(B * T, N, D)
            vit_image_features = self.vit(images)

            features = []
            for layer in self.vit_layers:
                features.append(vit_image_features[layer])
            vit_features = torch.cat(features, dim=-1).to(device=device)
            vit_feature_dim = vit_features.shape[-1]

            # Gather the features that should be pooled to build patch embeddings
            vit_features = vit_features.reshape(batch_size, -1, vit_feature_dim)[batch_idx[:, None, None], torch.clip(token_pooling, 0)]
            vit_features = vit_features * (token_pooling >= 0).float()[:, :, :, None]
            vit_features_mask = token_pooling >= 0

            # Build the sparse version which will be passed to the connector
            # Now shape [num_image_tokens_in_batch, pooling_dim, dim]
            image_features_mask = torch.any(vit_features_mask, -1)
            vit_features_flat = vit_features.reshape([-1, token_pooling.shape[-1], vit_features.shape[-1]])
            vit_features_flat = vit_features_flat[image_features_mask.view(-1)]
            vit_features_to_flat_mask = vit_features_mask.view(-1, token_pooling.shape[-1])[image_features_mask.view(-1)]

            # Finally, apply the connector and add to input embeddings
            image_features = self.connector(vit_features_flat, vit_features_to_flat_mask).to(device=device)
            x = x.clone()
            x.view(-1, dim)[is_image_token.view(-1)] += image_features.view(-1, dim)
        else:
            is_image_token = None
            is_indexable_image_token = None
            if image_data is not None:
                # Get the features/masks from the cache
                token_pooling = image_data.token_pooling.to(device=device)
                vit_features_mask = token_pooling >= 0
                image_features_mask = torch.any(vit_features_mask, -1)
                vit_features = image_data.vit_features.to(device=device)
            else:
                vit_features = None
                vit_features_mask = None
                image_features_mask = None

        # Embed the points
        if can_point:
            image_token_offset = image_data.flat_image_tokens_to_flat_image_features
            should_embed = (input_patch_ids >= 0) and (input_patch_ids < (bounds.patch_end-1))
            input_patch_ids_flat = (input_patch_ids + image_token_offset).view(-1)[should_embed.view(-1)]
            x.view(-1, dim)[is_patch.view(-1)] += image_data.image_features0.view(-1, dim)[input_patch_ids_flat]

            if torch.any(is_subpatch):
                vit_features_flat = vit_features.reshape([-1, token_pooling.shape[-1], vit_features.shape[-1]])
                vit_features_flat = vit_features_flat[image_features_mask.view(-1)]

                assert last_predicted_patch_id is not None, "Patch should always be generated before a subpatch"
                for_patches = (last_predicted_patch_id.view(batch_size) + image_token_offset)[input_subpatch_ids.view(batch_size) >= 0]
                vit_features_to_embed = vit_features_flat[for_patches, input_subpatch_ids]
                x.view(-1, dim)[is_subpatch.view(-1)] = self.build_vit_embedding(vit_features_to_embed).to(device=device, dtype=x.dtype)

        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # NOTE: this `is_prefill` logic is not flawless, it fails when we're using a cache eagerly initialized
        # (e.g. compiled prefill) AND `images` are not provided. Determining prefill in that case requires
        # checking data values, which is not compile-compatible.
        is_prefill = (
            not use_cache
            or past_key_values is None
            or not past_key_values.is_initialized
            or images is not None
        )

        # Adapted from transformers.models.gemma3.modeling_gemma3
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "input_embeds": x,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            if token_type_ids is not None and is_prefill:
                # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device)
                )

            # Create the mask
            causal_mask_mapping = create_causal_mask(**mask_kwargs)

        outputs = self.transformer(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            output_pre_ln_state=True,
            **kwargs,
        )
        x = outputs.pre_ln_hidden_state
        patch_logits = None
        subpatch_logits = None
        location_logits = None

        if images is not None or image_data is not None:
            patch_logits, subpatch_logits, location_logits, image_data = self.point_predictor(
                x,
                token_pooling,
                is_image_token,
                is_patch,
                is_subpatch,
                is_indexable_image_token,
                vit_features,
                vit_features_mask,
                image_features_mask,
                input_patch_ids,
                last_predicted_patch_id,
                image_data
            )
            if images is not None:
                # Also cache stuff we need to building the patch/subpatch token embeddings
                image_data.image_features0 = image_features
                num_image_tokens = is_image_token.sum(-1)
                image_token_offset = torch.cumsum(num_image_tokens[:-1], 0)
                image_token_offset = F.pad(image_token_offset, [1, 0])
                image_data.flat_image_tokens_to_flat_image_features = image_token_offset

        if last_predicted_patch_id is not None:
            last_predicted_patch_id = torch.where(input_patch_ids == -1, last_predicted_patch_id, input_patch_ids)
        else:
            last_predicted_patch_id = input_patch_ids

        return MolmoPointModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if images is not None else None,
            image_data=image_data,
            patch_logits=patch_logits,
            subpatch_logits=subpatch_logits,
            location_logits=location_logits,
            last_predicted_patch_id=last_predicted_patch_id,
        )


class ExtendedLmHead(nn.Module):
    def __init__(self, config, output_embeddings=None, new_output_embeddings=None):
        super().__init__()
        if output_embeddings is None:
            self.output_embeddings = nn.Parameter(torch.zeros([config.vocab_size, config.hidden_size]))
            self.new_output_embeddings = nn.Parameter(torch.zeros([128, config.hidden_size]))
        else:
            self.output_embeddings = output_embeddings
            self.new_output_embeddings = new_output_embeddings

    def __call__(self, hidden_states, slice_indices=None):
        lm_head = torch.concatenate([self.output_embeddings, self.new_output_embeddings], dim=0)
        return F.linear(hidden_states[:, slice_indices, :], lm_head.to(device=hidden_states.device))


class MolmoPointForConditionalGeneration(MolmoPointPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MolmoPointConfig

    def __init__(self, config: MolmoPointConfig):
        super().__init__(config)

        self.model = MolmoPointModel(config)
        if config.text_config.tie_word_embeddings:
            assert isinstance(self.model.transformer.wte, Molmo2Embedding)
            self.lm_head = ExtendedLmHead(config, self.model.transformer.wte.embedding, self.model.transformer.wte.new_embedding)
        else:
            self.lm_head = ExtendedLmHead(config)
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def _tied_weights_keys(self):
        if self.config.text_config.tie_word_embeddings:
            return ["lm_head.output_embeddings", "lm_head.new_output_embeddings"]
        return []

    def build_logit_processor_from_inputs(self, inputs) -> LogitsProcessorList:
        if inputs.get("image_token_pooling") is not None:
            pooling = inputs["image_token_pooling"]
        elif inputs.get("video_token_pooling") is not None:
            pooling = inputs["video_token_pooling"]
        else:
            return []
        return [self.build_logit_processor(pooling)]

    def build_logit_processor(self, token_pooling):
        return MolmoPointLogitProcessor(
            bounds=self.model.build_token_bounds(token_pooling),
            prevent_repeats=self.config.mask_repeats in ["all", "inference"],
            force_patch_sorted=self.config.mask_patches in ["always", "inference"],
            force_subpatch_sorted=self.config.mask_subpatches in ["always", "inference"],
        )

    def extract_image_points(self, output_text, pooling, subpatch_mapping, image_sizes):
        return extract_image_points(
            output_text, pooling, subpatch_mapping, self.config.no_more_points_class,
            self.config.patch_location, image_sizes)

    def extract_video_points(self, output_text, pooling, subpatch_mapping, timestamps, video_size):
        return extract_video_points(
            output_text, pooling, subpatch_mapping, timestamps, self.config.no_more_points_class,
            self.config.patch_location, video_size)

    def tie_weights(self):
        if self.config.text_config.tie_word_embeddings:
            self.lm_head.output_embeddings = self.model.transformer.wte.embedding
            self.lm_head.new_output_embeddings = self.model.transformer.wte.new_embedding

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.transformer.wte = value

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Make modules available throught conditional class for BC
    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.transformer

    @property
    def vision_backbone(self) -> torch.nn.Module:
        return self.model.vision_backbone

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_data: Optional[ImageCache] = None,
        last_predicted_patch_id: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MolmoPointCausalLMOutputWithPast]:
        r"""
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MolmoPointForConditionalGeneration

        >>> model = Molmo2ForConditionalGeneration.from_pretrained("...")
        >>> processor = AutoProcessor.from_pretrained("...")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
        ```"""
        outputs: MolmoPointModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            image_data=image_data,
            last_predicted_patch_id=last_predicted_patch_id,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states, slice_indices=slice_indices)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        bs, seq, _ = logits.shape
        if image_data is not None:
            token_pooling = image_data.token_pooling
        else:
            token_pooling = video_token_pooling if video_token_pooling is not None else image_token_pooling
        n_patches, n_subpatches = token_pooling.shape[-2:]
        if self.config.no_more_points_class:
            n_patches += 1
        small_val = -100000

        # The patch token is a bit tricky since we train the model to first select whether to
        # generate a patch token or not, and then to select the patch, but this two-stage
        # process is hard to emulate in generation frameworks
        # Our hack here is to assume that, if we generate a TOKEN, we always select the argmax
        # patch. Then we can use PATCH_TOKEN scores as the argmax's patch scores
        device = logits.device
        predicted_tokens = torch.argmax(logits[:, -1], dim=-1)
        patch_token_logits = torch.clone(logits[:, :, self.config.patch_token_id])
        logits[:, :, self.config.patch_token_id] = small_val
        predicted_patch = predicted_tokens == self.config.patch_token_id
        argmax_patch_logits = torch.full([bs, seq, n_patches], small_val, dtype=logits.dtype, device=device)
        if outputs.patch_logits is not None:
            selected_patches = torch.argmax(outputs.patch_logits, -1).to(device=device)
            bs, seq, n_patches = outputs.patch_logits.shape
            batch_idx = torch.arange(outputs.patch_logits.shape[0], device=device)
            seq_ix = torch.arange(outputs.patch_logits.shape[1], device=device)
            argmax_patch_logits[batch_idx.view(-1, 1, 1), seq_ix.view(1, -1, 1), selected_patches] = patch_token_logits

        logits[:, :, self.config.subpatch_token_id] = small_val
        if outputs.subpatch_logits is not None:
            subpatch_logits = outputs.subpatch_logits
        else:
            subpatch_logits = torch.full([bs, seq, n_subpatches], small_val, dtype=logits.dtype, device=device)

        logits[:, :, self.config.location_token_id] = small_val
        if outputs.location_logits is not None:
            location_logits = outputs.location_logits
        else:
            location_logits = torch.full([bs, seq, 9], small_val, dtype=logits.dtype, device=device)

        logits = torch.concatenate([
            logits,
            argmax_patch_logits,
            subpatch_logits.to(device=device),
            location_logits.to(device=device)
        ], -1)

        return MolmoPointCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            image_data=outputs.image_data,
            patch_logits=outputs.patch_logits,
            subpatch_logits=outputs.subpatch_logits,
            location_logits=outputs.location_logits,
            last_predicted_patch_id=outputs.last_predicted_patch_id,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        image_data: Optional[ImageCache] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            image_data=image_data,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: MolmoPointModelOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        args = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        if outputs.image_data is not None:
            args["image_data"] = outputs.image_data
        args["last_predicted_patch_id"] = outputs.last_predicted_patch_id
        return args

    # Adapted from transformers.models.gemma3.modeling_gemma3
    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device)
            )

        return create_masks_for_generate(**mask_kwargs)


# Always register for multi-modal features
AutoModelForImageTextToText.register(MolmoPointConfig, MolmoPointForConditionalGeneration)
