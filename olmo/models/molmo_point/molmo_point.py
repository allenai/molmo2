import dataclasses
import logging
import math
from dataclasses import field
from typing import (
    ClassVar,
    Optional,
    Sequence,
    Tuple,
    Iterator,
    List,
    Dict,
    Union,
)

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn import init

from olmo import tokenizer
from olmo.config import D, StrEnum
from olmo.data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from olmo.models.model import (
    OLMoOutput,
    OLMoGenerateOutput,
    ModelBase,
)
from olmo.models.model_config import BaseModelConfig
from olmo.models.molmo_point.modules import FlatRotaryEmbedding, PadWithLearnedVector
from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter
from olmo.models.molmo_point.molmo_point_example_preprocessor import \
    MolmoPointExamplePreprocessor, NO_POINTS_LABEL
from olmo.models.molmo_point.molmo_point_connector import ConnectorConfig
from olmo.models.molmo_point.molmo_point_text_preprocessor import MolmoPointTextPreprocessorConfig
from olmo.nn.beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from olmo.nn.cp_load_balancer import CPLoadBalancer
from olmo.nn.image_vit import VisionTransformer, VitConfig
from olmo.nn.legacy_config import convert_legacy_config
from olmo.nn.llm import LlmConfig, Llm, RMSLayerNorm
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.multimodal_collator import MMCollator
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
from olmo.tokenizer import get_special_token_ids
from olmo.torch_util import BufferCache, get_default_device, get_global_rank
from olmo.torch_util import collect_valid

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ImageCache:
    """Extra stuff we need to cache when doing autoregressive generation"""

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

    total_image_tokens: Optional[torch.LongTensor] = None
    """Cached for indexing uses"""


class MaskTokens(StrEnum):
    never = "never"
    always = "always"
    inference = "inference"


class RotaryType(StrEnum):
    none = "none"
    one_d = "one_d"
    one_d_last_two = "one_d_last_two"
    twh = "twh"
    t_wh_ordered = "t_wh_ordered"

    @classmethod
    def uses_3d_pos_ids(cls):
        return cls in [RotaryType.t_wh_ordered, RotaryType.twh]


class TokenCopy(StrEnum):
    modify_kv_cache = "modify_kv_cache"
    cache_hidden_states = "cache_hidden_states"


@dataclasses.dataclass
class MolmoPointPreprocessorConfig(MolmoPointTextPreprocessorConfig):
    """All preprocessing config options"""
    video: Optional[VideoPreprocessorConfig] = dataclasses.field(default_factory=VideoPreprocessorConfig)
    image: MultiCropConfig = dataclasses.field(default_factory=MultiCropConfig)
    remove_repeats: Optional[str] = None

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = super().update_legacy_settings(config)
        if config.video is not None:
            video_cfg = config.video
            if "use_col_tokens" in video_cfg:
                assert not video_cfg.pop("use_col_tokens")
            config.video = VideoPreprocessorConfig.update_legacy_settings(config.video)
        config.image = MultiCropConfig.update_legacy_settings(config.image)
        return config


@dataclasses.dataclass
class MolmoPointConfig(BaseModelConfig):
    """MolmoPoint model configuration"""
    _model_name: ClassVar[str] = "molmo_point"

    data_formatter: MolmoPointDataFormatter = field(default_factory=MolmoPointDataFormatter)
    """How to prompt the model for different tasks"""

    llm: LlmConfig = field(default_factory=LlmConfig)

    vit: Optional[VitConfig] = field(default_factory=VitConfig)
    """Builds patch-level features from input crops"""

    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    """Map the ViT patch-feature to features for the LLM"""

    mm_preprocessor: MolmoPointPreprocessorConfig = field(default_factory=MolmoPointPreprocessorConfig)
    """How to crop images and encoding jointly with text"""

    bi_directional_attn: Optional[str] = None
    """Allow bidirectional attention for some tokens"""

    shared_low_high_embedding: bool = True
    """Share initial embedding indexable and non-indexable image features"""

    patch_location: Optional[str] = "3x3"
    """How to include within patch location predictions"""

    no_more_points_class: bool = False
    """Include a no-more-point class when cross-attending to image tokens"""

    patch_embed_dim: int = 256
    """"Dim to use when selecting image tokens or ViT patches"""

    patch_embedding_kind: str = "linear"
    """How to embed the selected image token into the model's input"""

    embed_selected_vit_patch: Optional[str] = "linear"
    """How to embed the selected Vit patch into the model's input"""

    embed_location: Optional[bool] = False
    """Whether to embed the selected location as in the model's input"""

    layer_norm_x: bool = True
    """Apply a layer not to the hidden state before cross-attending"""

    norm_x: bool = True
    """Apply square root normalization to the hidden state (overriden by layer_norm_x)"""

    norm_logits: bool = True
    """Norm token/patch logits by 1/sqrt(patch_embed_dim)"""

    mask_patches: Optional[MaskTokens] = MaskTokens.always
    """When to mask tokens that are before the previously selected token"""

    sort_points: bool = True
    """Sort points"""

    mask_subpatches: MaskTokens = MaskTokens.inference
    """When to mask ViT patches that are before the previously selected ViT patch"""

    mask_repeats: Optional[MaskTokens] = MaskTokens.inference
    """When to prevent the model selecting the same ViT patch twice"""

    token_prediction_rotary: RotaryType = RotaryType.one_d
    """How to rotate token keys/queries"""

    token_prediction_rotary_theta: Optional[float] = 50000
    """Theta for token keys/queries rotations"""

    token_prediction_rotary_dims: Optional[List[int]] = None
    """How to split up position ids if using 3D rotations"""

    debug: Optional[str] = None
    """Just for debugging purposes"""

    def __post_init__(self):
        super().__post_init__()
        self.data_formatter._location_token = (self.patch_location is not None)
        self.data_formatter._end_with_patch = self.no_more_points_class

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        for k in [
            "cross_attend_layers",
            "image_token_copy",
            "cp_enabled",
            "point_start_token",
            "apply_cp_to_vision_backbone",
        ]:
            if k in config:
                assert not config.pop(k)
        if "cross_attend_dim" in config:
            config.pop("cross_attend_dim")
        if "layers_to_add_image_features_to" in config:
            assert config.pop("layers_to_add_image_features_to") == [0]
        config.llm = LlmConfig.update_legacy_settings(config.llm)
        config.connector = ConnectorConfig.update_legacy_settings(config.connector)
        config.data_formatter = MolmoPointDataFormatter.update_legacy_settings(config.data_formatter)
        config.mm_preprocessor = MolmoPointPreprocessorConfig.update_legacy_settings(config.mm_preprocessor)
        return config

    def build_tokenizer(self):
        """Tokenizer this model uses"""
        return self.llm.build_tokenizer()

    def build_preprocessor(
        self,
        for_inference,
        is_training=True,
        text_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        include_image=False
    ) -> MolmoPointExamplePreprocessor:
        """
        Build a preprocessor that converts 'raw' image/text data from various tasks into tensors
        inputs/targets that can be passed to the model's forward/generate methods
        """
        if self.token_prediction_rotary == RotaryType.t_wh_ordered:
            image_pos_ids = "t-wh-ordered"
        elif self.token_prediction_rotary == RotaryType.twh:
            image_pos_ids = "twh"
        else:
            image_pos_ids = "one_d"
        tok = self.build_tokenizer()
        image_preprocessor = self.connector.build_preprocessor(self.vit)
        image, multi_image = self.mm_preprocessor.image.build_image_preprocessor(
            tok, image_preprocessor, None, use_low_res_token_global_crops=True)
        text = self.mm_preprocessor.build_text_preprocessor(tok, max_seq_len)
        video = None
        if self.mm_preprocessor.video is not None:
            video = self.mm_preprocessor.video.build_video_preprocessor(tok, image_preprocessor)
        return MolmoPointExamplePreprocessor.build(
            data_formatter=self.data_formatter,
            for_inference=for_inference,
            is_training=is_training,
            text_preprocessor=text,
            image_preprocessor=image,
            multi_image_preprocessor=multi_image,
            video_preprocessor=video,
            patch_location=self.patch_location,
            image_pos_ids=image_pos_ids,
            text_seq_len=text_seq_len,
            remove_repeats=self.mm_preprocessor.remove_repeats,
            include_image=include_image,
            end_of_group_target_id=self.no_more_points_class,
            sort_points=self.sort_points

        )

    def token_ids_to_coordinates(self, text, target_ids, metadata: Dict):
        """Convert patch/location ids generated by the model to pixel coordinates

        text: str string output for the model
        target_ids: [n_points, 2] or [n_points, 3] Target ids produced by `generate`
        metadata: Dict metadata from the preprocessor

        output: A [n_points 3] (for image) or [n_points, 4] (for video/multi-image) float array
                with [object_id, x, w] or [object_id, t, x, y] coordinates
        """
        return MolmoPointExamplePreprocessor.token_ids_to_coordinates(text, target_ids, metadata, self.patch_location)

    def build_collator(self, output_shapes, pad_mode: str, include_metadata=True) -> MMCollator:
        """Collators for tensors the preprocessor produces"""
        return MMCollator(
            get_special_token_ids(self.build_tokenizer()),
            output_shapes,
            include_metadata=include_metadata,
            pad=pad_mode,
        )

    def build_model(self, device=None):
        return MolmoPoint(self, device)

    @property
    def max_sequence_length(self):
        return self.llm.max_sequence_length


def _weighted_cross_entropy(logits, targets, weight, mask, labels_padded=False):
    with torch.autocast(enabled=False, device_type=logits.device.type):
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        logits = logits.float()
        if weight is None:
            return F.cross_entropy(logits, targets, reduction='sum', ignore_index=-1)
        else:
            losses = F.cross_entropy(logits, targets, reduction='none', ignore_index=-1)
            if labels_padded:
                losses = losses[targets != -1]
            return torch.dot(losses, weight.view(-1)[mask.view(-1)].float())


class MolmoPoint(ModelBase):

    def __init__(self, config: MolmoPointConfig, device=None):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        self.transformer: Llm = self.config.llm.build(self.__cache, device)
        assert self.config.llm.can_predict_extra_tokens or self.config.llm.weight_tying
        self.connector = self.config.connector.build(self.config.llm, self.config.vit, device)

        self.vit_layers = []
        for layer in config.connector.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.vit.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        vit_cfg = self.config.vit
        if last_layer_needed < config.vit.image_num_layers:
            vit_cfg = dataclasses.replace(vit_cfg, image_num_layers=last_layer_needed)
            self.vit: VisionTransformer = vit_cfg.build(device)
        else:
            self.vit: VisionTransformer = vit_cfg.build(device)
        self.image_preprocessor = self.config.connector.build_preprocessor(self.config.vit)

        d_vit = self.config.vit.image_emb_dim * len(self.config.connector.vit_layers)
        pooling_size = config.mm_preprocessor.image.pooling_w * config.mm_preprocessor.image.pooling_h
        if config.mm_preprocessor.video is not None:
            pooling_size = max(
                pooling_size,
                config.mm_preprocessor.video.pooling_w * config.mm_preprocessor.video.pooling_h
            )

        if self.config.patch_embedding_kind in ["image_feature0"]:
            self.build_pointing_embedding = None
        else:
            input_dim = d_vit*pooling_size
            self.build_pointing_embedding = nn.Linear(input_dim, self.config.llm.d_model, device=device)

        if self.config.embed_selected_vit_patch == "linear_with_pos":
            self.build_vit_embedding = nn.Linear(d_vit, self.config.llm.d_model, device=device, bias=True)
            self.vit_pos_embed = nn.Embedding(pooling_size, self.config.llm.d_model, device=device)
        elif self.config.embed_selected_vit_patch == "linear":
            self.build_vit_embedding = nn.Linear(d_vit, self.config.llm.d_model, device=device, bias=True)
            self.vit_pos_embed = None
        elif self.config.embed_selected_vit_patch is None:
            self.vit_pos_embed = None
            self.build_vit_embedding = None
        else:
            raise NotImplementedError(f"Embedding {self.config.embed_selected_vit_patch} not implemented")

        if self.config.embed_location:
            assert self.config.patch_location == "3x3"
            self.loc_pos_embed = nn.Embedding(9, self.config.llm.d_model, device=device)
        else:
            self.loc_pos_embed = None

        if self.config.layer_norm_x:
            self.x_norm = RMSLayerNorm(self.config.llm)
        else:
            self.x_norm = None

        project_fn = lambda _dim: nn.Linear(_dim, config.patch_embed_dim, device=device)
        self.patch_q = project_fn(self.config.llm.d_model)
        self.patch_k = project_fn(self.config.llm.d_model)
        self.subpatch_q = project_fn(self.config.llm.d_model)
        self.subpatch_k = project_fn(d_vit)

        if self.config.no_more_points_class:
            self.add_no_point_class_embed = PadWithLearnedVector(config.patch_embed_dim)
        else:
            self.add_no_point_class_embed = None

        if self.config.patch_location == "3x3":
            self.subpatch_loc_k = nn.Linear(self.config.llm.d_model, 9, device=device)
        elif self.config.patch_location is not None:
            raise ValueError()
        else:
            self.subpatch_loc_k = None

        if self.config.token_prediction_rotary == RotaryType.none:
            self.patch_rotary = None
        else:
            theta = self.config.token_prediction_rotary_theta or self.config.llm.rope_theta
            if self.config.token_prediction_rotary == RotaryType.one_d:
                self.patch_rotary = FlatRotaryEmbedding(
                    theta, self.config.patch_embed_dim, "patch", self.__cache, self.config.llm.max_sequence_length, device)
            elif self.config.token_prediction_rotary == RotaryType.one_d_last_two:
                self.patch_rotary = FlatRotaryEmbedding(
                    theta, self.config.token_prediction_rotary_dims, "patch", self.__cache, self.config.llm.max_sequence_length, device)
            elif self.config.token_prediction_rotary in [RotaryType.t_wh_ordered, RotaryType.twh]:
                self.patch_rotary = FlatRotaryEmbedding(
                    theta, self.config.token_prediction_rotary_dims, "patch", self.__cache, self.config.llm.max_sequence_length, device)
            else:
                raise NotImplementedError()

        self.special_ids = tokenizer.get_special_token_ids(self.config.build_tokenizer())
        if self.config.bi_directional_attn:
            self.__cache["image_tokens"] = torch.as_tensor([self.special_ids[x] for x in [
                tokenizer.IMAGE_PATCH_TOKEN,
                tokenizer.IM_COL_TOKEN,
                tokenizer.IM_START_TOKEN,
                tokenizer.LOW_RES_IMAGE_START_TOKEN,
                tokenizer.FRAME_START_TOKEN,
                tokenizer.IM_END_TOKEN,
                tokenizer.FRAME_END_TOKEN,
                tokenizer.IMAGE_LOW_RES_TOKEN,
            ]], dtype=torch.long, device=get_default_device())
        self._low_res_image_start = self.special_ids[tokenizer.LOW_RES_IMAGE_START_TOKEN]
        self._frame_end = self.special_ids[tokenizer.FRAME_END_TOKEN]
        self._frame_start = self.special_ids[tokenizer.FRAME_START_TOKEN]
        self._image_end_token_id = self.special_ids[tokenizer.IM_END_TOKEN]
        self._image_start_token_id = self.special_ids[tokenizer.IM_START_TOKEN]
        self._image_low_res_id = self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN]
        self._image_high_res_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]
        self._image_patch_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]
        self._image_col_token_id = self.special_ids[tokenizer.IM_COL_TOKEN]
        self._target_patch_token = self.special_ids[tokenizer.TOKEN_INDEX_TOKEN]
        self._target_subpatch_token = self.special_ids[tokenizer.SUBPATCH_INDEX_TOKEN]
        self._target_subpatch_location = self.special_ids[tokenizer.LOCATION_CLS_TOKEN]

        # FIXME this is a bit of hack: We don't have an official "point start" token, so we
        # use starting text id. This works because a point group only has digits/special tokens
        # so we can assume that any points after it, but before the next occurance, are sorted
        self._single_point_seq = self.config.build_tokenizer().encode("<points coords=\"<|token_index|><|vit_index|><|vit_loc|>1<|token_index|>\">")
        self._point_start_id = self.config.build_tokenizer().encode("coords=\"")[-1]
        tmp = self.config.build_tokenizer().encode(" 1")
        assert len(tmp) == 2
        self._space_token = tmp[0]

    def get_legacy_key_mapping(self):
        return {f"connector.{k}": f"connectors.0.{k}" for k in self.connector.state_dict().keys()}

    def get_pointing_modules(self):
        for mod in [
            self.patch_q, self.patch_k, self.subpatch_k, self.subpatch_q, self.subpatch_loc_k,
            self.build_vit_embedding, self.build_pointing_embedding, self.vit_pos_embed,
            self.x_norm, self.loc_pos_embed, self.add_no_point_class_embed
        ]:
            if mod is not None:
                yield mod

    def reset_parameters(self):
        """Re-initialize the weights from scratch"""
        self.transformer.reset_parameters()
        self.vit.reset_parameters()
        self.connector.reset_parameters()
        self._reset_point_predictors()

    def reset_with_pretrained_weights(self):
        """Re-initialize the weights, possibly loading pretrained weights for the LLM and ViT"""
        self.transformer.reset_with_pretrained_weights()
        self.vit.reset_with_pretrained_weights()
        self.connector.reset_parameters()
        self._reset_point_predictors()

    def _reset_point_predictors(self):
        for mod in self.get_pointing_modules():
            if isinstance(mod, (RMSLayerNorm, PadWithLearnedVector)):
                mod.reset_parameters()
            elif isinstance(mod, nn.Embedding):
                init.zeros_(mod.weight)
            else:
                if mod.bias is not None:
                    init.zeros_(mod.bias)
                init.normal_(mod.weight, 0, 0.02)

    def apply_activation_checkpointing(self):
        """Enable activation checkpointing"""
        self.transformer.apply_activation_checkpointing()
        self.vit.apply_activation_checkpointing()
        self.connector.apply_activation_checkpointing()

    def apply_compile(self, **compile_kwargs):
        """Compile the model with `torch.compile`"""
        self.transformer.apply_compile(**compile_kwargs)
        for block in self.vit.transformer.resblocks:
            block.compile(**compile_kwargs)
        self.connector.apply_compile(**compile_kwargs)

    def warmup_cache(self, device, cp_enabled: bool = False):
        """Pre-fill the buffer-cache"""
        if self.transformer.blocks[0].rotary_emb is not None:
            self.transformer.blocks[0].rotary_emb.warmup_cache(device, cp_enabled=cp_enabled)
        if self.patch_rotary is not None:
            self.patch_rotary.warmup_cache(device, cp_enabled=cp_enabled)

    def apply_fsdp2(self, **fully_shard_kwargs):
        """Fully shard this model using `fully_shard`"""
        self.transformer.apply_fsdp2(**fully_shard_kwargs)
        self.connector.apply_fsdp2(**fully_shard_kwargs)
        self.vit.apply_fsdp2(**fully_shard_kwargs)

        qk = [self.patch_q, self.patch_k, self.subpatch_k, self.subpatch_q,
              self.x_norm, self.subpatch_loc_k, self.add_no_point_class_embed]
        fully_shard([x for x in qk if x is not None], **fully_shard_kwargs)

        embed = [self.build_pointing_embedding, self.build_vit_embedding, self.vit_pos_embed, self.loc_pos_embed]
        fully_shard([x for x in embed if x is not None], **fully_shard_kwargs)
        fully_shard_kwargs = dict(fully_shard_kwargs)
        fully_shard_kwargs["mp_policy"] = dataclasses.replace(fully_shard_kwargs['mp_policy'], cast_forward_inputs=False)
        fully_shard(self, **fully_shard_kwargs)

    def get_frame_selection_parameters(self) -> Union[List, Iterator[torch.Tensor]]:
        parameters = []
        for mod in self.get_pointing_modules():
            parameters += list(mod.parameters())
        return parameters

    def get_connector_parameters(self) -> Iterator[torch.Tensor]:
        parameters = list(self.connector.parameters())
        if self.config.llm.additional_vocab_size:
            parameters.append(self.transformer.wte.new_embedding)
        if not self.config.llm.weight_tying:
            parameters.append(self.transformer.ff_out.new_weight)
        return parameters

    def get_vit_parameters(self) -> Iterator[torch.Tensor]:
        return self.vit.parameters()

    def get_llm_parameters(self) -> Iterator[torch.Tensor]:
        c_params = set(self.get_connector_parameters())
        return [p for p in self.transformer.parameters() if p not in c_params]

    def get_non_weight_decay_params(self) -> Iterator[torch.Tensor]:
        exclude_list = {
            "wte", "attn_norm", "ff_norm",
            "pre_attn_norm", "post_attn_norm",
            "pre_ff_norm", "post_ff_norm",
            "ln_f",
            "pre_ln",
            "attention_norm", "ffn_norm",
            "lambda1", "lambda2",
            "positional_embedding", "class_embedding", "patch_embedding",
        }
        return (param for name, param in self.named_parameters() if
                any(part in exclude_list for part in name.split(".")))

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """Get the total number of parameters."""
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        if not include_inactive_params:
            # Need to reduce blocks to the number of experts that are selected
            # If not dropless 'transformer.blocks.0.ffn.experts.mlp.w1' has shape (total_experts, in_dim, out_dim)
            # change to 'transformer.blocks.0.ffn.experts.mlp.w1' with shape (selected_experts, in_dim, out_dim)
            # If dropless, the total_experts & out_dim are combined into one dimension
            idx = self.config.llm.moe_top_k
            if self.config.llm.moe_dropless:
                idx *= self.transformer.blocks[1].moe_args.ffn_hidden_size
            params = [(np[0], np[1][:idx]) if "experts.mlp" in np[0] else np for np in params]  # type: ignore
        return sum(p.numel() for _, p in params)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        response_logits_only = False,
        point_target_ids: Optional[torch.Tensor] = None,

        # Image data
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_pos_ids: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,

        # For inference
        prev_target_ids: Optional[torch.Tensor] = None,

        # Generation args
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        image_data: Optional[ImageCache]= None,

        **kwargs,
    ) -> OLMoOutput:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.llm.n_layers

        has_image = images is not None

        assert not (
            has_image and input_embeddings is not None
        ), "Cannot provide both images and input embeddings."
        assert not (
            has_image and past_key_values is not None
        ), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        dim = self.config.llm.d_model
        dev = input_ids.device
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Build position_ids and attention_mask if needed
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = input_ids != -1
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            if position_ids is None:
                if subsegment_ids is not None:
                    raise ValueError(f"Positioned ids must be given if using subsegment_ids")
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                    ).broadcast_to((batch_size, attention_mask.shape[-1]))
        else:
            assert attention_mask is not None
            assert position_ids is not None

        # Transform the attention mask into a 3D tensor
        attention_mask_len = past_length + seq_len  # mask should include the K/V cache
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, :attention_mask_len]
            attention_mask = attention_mask[:, None, :]
        assert attention_mask.shape[-1] == attention_mask_len

        # Build casual mask
        if "casual_mask" not in self.__cache or self.__cache["casual_mask"].shape[-1] < attention_mask_len:
            self.__cache["casual_mask"] = torch.tril(torch.ones(
                attention_mask_len, attention_mask_len,device=dev, dtype=torch.bool))[None, :, :]
        casual_mask = self.__cache["casual_mask"].to(dev)[:, :attention_mask_len, :attention_mask_len]

        # Modify to allow select bi-directional attention if configured
        bidir_mask = None
        if self.config.bi_directional_attn == "image_tokens":
            is_image_token = self.__cache["image_tokens"].to(input_ids.device)
            c = torch.any(input_ids[:, :, None] == is_image_token[None, None, :], -1)
            bidir_mask = (c[:, :, None] & c[:, None, :])
        elif self.config.bi_directional_attn == "within_image":
            # Important! this assumes self._low_res_image_start is used to start images
            is_frame_start = (input_ids == self._frame_start) | (input_ids == self._low_res_image_start)
            frame_id = torch.cumsum(is_frame_start, dim=-1)
            same_frame = frame_id[:, None] <= frame_id[:, :, None]
            is_image_token = self.__cache["image_tokens"].to(input_ids.device)
            c = torch.any(input_ids[:, :, None] == is_image_token[None, None, :], -1)
            bidir_mask = (c[:, :, None] & c[:, None, :]) & same_frame
        elif self.config.bi_directional_attn == "image_to_question":
            if images is not None:
                # image tokens can attend to all non-response tokens
                is_image_token = self.__cache["image_tokens"].to(input_ids.device)
                is_image_token = torch.any(input_ids[:, :, None] == is_image_token[None, None, :], -1)
                if use_cache:
                    bidir_mask = is_image_token[:, :, None]
                else:
                    bidir_mask = (is_image_token[:, :, None] & (~response_mask[:, None, :]))
        elif self.config.bi_directional_attn is not None:
            raise NotImplementedError(self.config.bi_directional_attn)

        if bidir_mask is not None:
            if subsegment_ids is not None:
                example_id = subsegment_ids // EXAMPLE_SUBSEGMENT_INCREMENT
                bidir_mask = bidir_mask & (example_id[:, None] == example_id[:, :, None])
            attention_mask = attention_mask & (casual_mask | bidir_mask)
        else:
            attention_mask = attention_mask & casual_mask

        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = attention_mask & subsegment_mask

        attention_mask = attention_mask.unsqueeze(1)  # for head dimension

        if input_embeddings is not None:
            x = input_embeddings
        elif self.config.shared_low_high_embedding:
            x = self.transformer.wte(torch.where(input_ids == self._image_low_res_id, self._image_high_res_id, input_ids))
        else:
            x = self.transformer.wte(input_ids)

        # Convert mask to a float mask, and possibly combine with `attention_bias`
        if attention_bias is not None:
            attention_bias = torch.where(attention_mask, attention_bias, torch.finfo(x.dtype).min)
        else:
            attention_bias = torch.where(attention_mask, 0, torch.finfo(x.dtype).min)

        batch_idx = torch.arange(0, batch_size, device=dev)
        o = torch.ones((), dtype=x.dtype, device=dev)

        # The global-crop in images will use `self._image_low_res_id` to mark they should
        # not be indexed
        is_indexable_image_token = input_ids == self._image_high_res_id
        is_non_indexable_image_token = input_ids == self._image_low_res_id
        is_image_token = is_indexable_image_token | is_non_indexable_image_token

        # Extract the image features or load them from the cache
        if images is not None:
            cfg = self.config
            B, T, N, D = images.shape
            images = images.view(B * T, N, D)
            if cfg.connector.normalize_on_gpu:
                images = self.image_preprocessor.normalize_image_tensor(images)
            vit_layer_features = self.vit(images)
            features = []
            for layer in self.vit_layers:
                features.append(vit_layer_features[layer])
            vit_features = torch.cat(features, dim=-1)
            del vit_layer_features, features

            # Gather the features that should be pooled to build patch embeddings
            vit_features = vit_features.reshape(batch_size, -1, vit_features.shape[-1])[batch_idx[:, None, None], torch.clip(token_pooling, 0)]
            vit_features = vit_features * (token_pooling >= 0).float()[:, :, :, None]
            vit_features_mask = token_pooling >= 0

            # Build the sparse version which will be passed to the connector
            # Now shape [num_image_tokens_in_batch, pooling_dim, dim]
            image_features_mask = torch.any(vit_features_mask, -1)
            vit_features_flat = vit_features.reshape([-1, token_pooling.shape[-1], vit_features.shape[-1]])
            vit_features_flat = vit_features_flat[image_features_mask.view(-1)]
            vit_features_to_flat_mask = vit_features_mask.view(-1, token_pooling.shape[-1])[image_features_mask.view(-1)]

            # Finally apply the connector and add to input embeddings
            image_features = self.connector(vit_features_flat, vit_features_to_flat_mask)
            x = x.clone()
            x.view(-1, dim)[is_image_token.view(-1)] += image_features.view(-1, dim)
        else:
            token_pooling = image_data.token_pooling
            if token_pooling is not None:
                vit_features = image_data.vit_features
                vit_features_mask = token_pooling >= 0
                image_features_mask = torch.any(vit_features_mask, -1)
                vit_features_flat = vit_features.reshape([-1, token_pooling.shape[-1], vit_features.shape[-1]])
                vit_features_flat = vit_features_flat[image_features_mask.view(-1)]
                vit_features_to_flat_mask = vit_features_mask.view(-1, token_pooling.shape[-1])[image_features_mask.view(-1)]
                vit_features_to_flat_mask = vit_features_mask.view(-1, token_pooling.shape[-1])[image_features_mask.view(-1)]
            else:
                vit_features = None
                image_features_mask = None
                vit_features_mask = None
                vit_features_flat = None
                vit_features_to_flat_mask = None
            image_features = image_data.image_features0
            image_pos_ids = image_data.image_pos_ids

        doing_inference = past_key_values is not None
        doing_prefilling = append_last_valid_logits is not None

        # Compute offsets into the flatten image features, which we might need for
        # index manipulation
        num_image_tokens = None
        image_token_offset = None
        if image_data is not None:
            num_image_tokens = image_data.total_image_tokens  # might be cached
        elif doing_prefilling or point_target_ids is not None:
            num_image_tokens = is_image_token.sum(-1)

        if num_image_tokens is not None:
            image_token_offset = torch.cumsum(num_image_tokens[:-1], 0)
            image_token_offset = F.pad(image_token_offset, [1, 0])

        if point_target_ids is not None:
            # Embed the pointing predictions in the input embeddings

            is_patch_token = input_ids == self._target_patch_token
            target_patch_ids = point_target_ids[:, :, 0]
            if self.config.no_more_points_class:
                # only contains points that are not in the no-point class
                with_subpatch_patch_ids = torch.where(target_patch_ids == NO_POINTS_LABEL, -1, target_patch_ids)
            else:
                with_subpatch_patch_ids = target_patch_ids

            target_subpatch_ids = point_target_ids[:, :, 1]
            if doing_inference:
                valid = is_patch_token.view(-1)
                assert torch.all(target_patch_ids.view(-1)[valid] >= 0)
            else:
                valid = (target_patch_ids >= 0).view(-1)

            # Index of (flattened) predicted patches into the (flattened) image features
            if batch_size > 1:
                point_target_patch_ids_flat = target_patch_ids + image_token_offset[:, None]
            else:
                point_target_patch_ids_flat = target_patch_ids
            point_target_patch_ids_flat = point_target_patch_ids_flat.view(-1)[valid]

            if self.build_pointing_embedding is not None:
                assert not self.config.no_more_points_class
                flat_dim = self.build_pointing_embedding.weight.shape[1]
                selected_vit_features = vit_features_flat[point_target_patch_ids_flat]
                selected_vit_features = selected_vit_features.view(point_target_patch_ids_flat.shape[0], flat_dim)
                point_features = self.build_pointing_embedding(selected_vit_features)
            elif self.config.patch_embedding_kind == "image_feature0":
                if self.config.no_more_points_class:
                    point_features = torch.where(
                        (point_target_patch_ids_flat < NO_POINTS_LABEL)[:, None],
                        image_features.view(-1, dim)[torch.clamp(point_target_patch_ids_flat, max=image_features.shape[0]-1)],
                        0,
                    )
                else:
                    point_features = image_features[point_target_patch_ids_flat]
            else:
                raise ValueError()
            x = x.clone()
            x.view(-1, dim)[is_patch_token.view(-1)] += point_features.view(-1, dim)

            if self.build_vit_embedding is not None:
                # subpatch tokens get an embedding based on the ViT patch
                is_subpatch_token = input_ids == self._target_subpatch_token
                if doing_inference:
                    valid_subpatch = is_subpatch_token.view(-1)
                    assert torch.all(point_target_ids[:, :, :2].view(-1, 2)[valid_subpatch] >= 0)
                else:
                    valid_subpatch = target_subpatch_ids.view(-1) >= 0
                selected_vit_patch = vit_features.reshape(batch_size, -1, vit_features.shape[-2], vit_features.shape[-1])
                selected_vit_patch = selected_vit_patch[batch_idx[:, None],  with_subpatch_patch_ids, target_subpatch_ids]
                selected_vit_patch = selected_vit_patch.view(-1, vit_features.shape[-1])
                selected_vit_patch = selected_vit_patch[valid_subpatch]
                selected_vit_features = self.build_vit_embedding(selected_vit_patch)
                if self.vit_pos_embed is not None:
                    selected_vit_features = selected_vit_features + self.vit_pos_embed(target_subpatch_ids.view(-1)[valid_subpatch])
                x = x.clone()
                x.view(-1, x.shape[-1])[is_subpatch_token.view(-1)] += selected_vit_features.to(dtype=x.dtype)

            if self.loc_pos_embed is not None:
                is_loc_token = input_ids == self._target_subpatch_location
                if doing_inference:
                    valid_loc = is_loc_token.view(-1)
                    assert torch.all(point_target_ids.view(-1, 3)[valid_loc] >= 0)
                else:
                    valid_loc = target_subpatch_ids.view(-1) >= 0
                loc_ids = point_target_ids[:, :, 2].view(-1)
                loc_embed = self.loc_pos_embed(loc_ids[valid_loc])
                x = x.clone()
                x.view(-1, x.shape[-1])[is_loc_token.view(-1)] += loc_embed.to(dtype=x.dtype)
        else:
            target_patch_ids = None

        if not self.config.llm.rope:
            raise NotImplementedError()

        x = self.transformer.emb_drop(x)

        if self.config.llm.normalize_input_embeds:
            raise NotImplementedError()

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        all_hidden_states = []
        selected_patches_ixs = None
        for block_idx, block in enumerate(self.transformer.blocks):
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            x, cache = block(
                x,
                attention_bias=attention_bias,
                position_ids=position_ids,
                drop_mask=response_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        metrics = {}
        if self.x_norm:
            x_norm = self.x_norm(x) 
        elif self.config.norm_x:
            x_norm = x / math.sqrt(dim)
        else:
            x_norm = x

        # Tokens that will predict a patch
        patch_predictor_tokens = F.pad((input_ids == self._target_patch_token)[:, 1:], [0, 1])
        any_patch_predictors = torch.any(patch_predictor_tokens)

        # Tokens that will predict a subpatch
        subpatch_predictor_tokens = F.pad((input_ids == self._target_subpatch_token)[:, 1:], [0, 1])
        any_subpatch_predictors = torch.any(subpatch_predictor_tokens)

        if image_pos_ids.shape[1] > 0:
            max_image_pos_id = image_pos_ids.max() + 1
        else:
            max_image_pos_id = 1

        # Build the keys, or get them from the cache
        if image_data is not None:
            patch_k, subpatch_k = image_data.patch_k, image_data.subpatch_k
            patch_k_mask = image_data.patch_k_mask
        else:
            patch_k_flat = self.patch_k(x_norm.view(-1, dim)[is_image_token.view(-1)])
            if self.patch_rotary is not None:
                image_pos_ids_flat = image_pos_ids.view(-1, image_pos_ids.shape[-1])[(image_pos_ids[:, :, 0] >= 0).view(-1)]
                patch_k_flat = self.patch_rotary(patch_k_flat, image_pos_ids_flat, max_len=max_image_pos_id)
            patch_k = torch.zeros([batch_size, image_features_mask.shape[1], patch_k_flat.shape[-1]], dtype=x.dtype, device=dev)
            patch_k.view(-1, patch_k_flat.shape[-1])[image_features_mask.flatten()] = patch_k_flat.to(dtype=x.dtype)

            patch_k_mask = image_features_mask.clone()
            patch_k_mask.view(-1)[image_features_mask.view(-1)] = (
                is_indexable_image_token.view(-1)[is_image_token.view(-1)])

            if self.config.no_more_points_class:
                patch_k = self.add_no_point_class_embed(patch_k)
                patch_k_mask = F.pad(patch_k_mask, (0, 1), value=True)

            subpatch_k = self.subpatch_k(vit_features)

        # Now we need to build the token queries from the hidden states of the pointing tokens
        # and use the to compute the loss or make predictions
        patch_logits, subpatch_logits, location_logits = None, None, None
        token_losses = []

        # Make patch predictions
        if doing_inference:
            # Always compute patch logits for the token just in case we end up
            # sampling a pointing prediction token
            assert seq_len == 1
            image_q = self.patch_q(x_norm)
            if self.patch_rotary is not None and prev_target_ids is not None:
                assert prev_target_ids.shape[1] == 1
                rotate_by = prev_target_ids[:, 0, 0]
                if self.config.no_more_points_class:
                    rotate_by = image_pos_ids[batch_idx, torch.clamp(prev_target_ids[:, 0, 0] ,0, max=image_pos_ids.shape[1]-1)]
                    rotate_by = rotate_by.squeeze(-1)
                    rotate_by = torch.where(prev_target_ids[:, 0, 0] == NO_POINTS_LABEL, 0, rotate_by)
                else:
                    rotate_by = image_pos_ids[batch_idx, torch.clamp(prev_target_ids[:, 0, 0] , 0)]
                    rotate_by = rotate_by.squeeze(-1)
                image_q = self.patch_rotary(
                    image_q.view(-1, image_q.shape[-1]),
                    torch.clamp(rotate_by, min=0),
                    max_len=max_image_pos_id
                ).reshape(batch_size, -1, image_q.shape[-1])

            dots = torch.matmul(image_q, patch_k.transpose(1, 2))  # [batch, 1, num_images]
            if self.config.norm_logits:
                dots = dots / math.sqrt(dots.shape[-1])

            valid = patch_k_mask[:, None, :]
            patch_logits = torch.where(valid, dots, -100000000)

        elif target_patch_ids is not None and any_patch_predictors:
            # Queries in flat format: [n_patch_tokens_in_batch, query_dim]
            image_q_flat = self.patch_q(x_norm.view(-1, dim)[patch_predictor_tokens.view(-1)])

            # Rotate
            num_valid = (target_patch_ids >= 0).sum(-1)
            max_valid = num_valid.max()
            if self.patch_rotary is not None and image_q_flat.shape[1] > 0:
                rotate_by = F.pad(with_subpatch_patch_ids[:, :max_valid-1], [1, 0], value=0)
                rotate_by = image_pos_ids[batch_idx[:, None], torch.clamp(rotate_by ,0)]
                rotate_by = rotate_by.view(-1)[(target_patch_ids[:, :max_valid] >= 0).view(-1)]
                image_q_flat = self.patch_rotary(image_q_flat, rotate_by, max_len=max_image_pos_id)

            # Unflatten into [batch, n_queries] format with a mask
            point_mask = torch.arange(0, max_valid, device=x.device)[None, :] < num_valid[:, None]
            image_q = torch.zeros([batch_size, max_valid, image_q_flat.shape[-1]], device=x.device, dtype=x.dtype)
            image_q.view(-1, image_q_flat.shape[-1])[point_mask.flatten()] = image_q_flat.to(dtype=x.dtype)

            # Compute the scores
            dots = torch.matmul(image_q, patch_k.transpose(1, 2))  # [batch, num_ponts, num_images]
            if self.config.norm_logits:
                dots = dots / math.sqrt(dots.shape[-1])
            valid = patch_k_mask[:, None, :] * point_mask[:, :, None]
            patch_logits = torch.where(valid, dots, -100000000)

            assert torch.all(target_patch_ids[:, patch_logits.shape[1]:] == -1)
            labels = target_patch_ids[:, :patch_logits.shape[1]]

            if self.config.mask_patches == MaskTokens.always:
                # Do masking, this is a slightly tricky since we need to handle multiple
                # sets of points in the sequence

                # Figure out which points belong to the same group of sorted points
                # We need to know this since the first point in a group should not
                # mask points in the previous group which have been sorted independently
                might_start_point_group = (input_ids == self._point_start_id)
                token_point_group_id = torch.cumsum(might_start_point_group, dim=-1)
                point_group_ids = torch.zeros([batch_size, labels.shape[1]], device=labels.device, dtype=token_point_group_id.dtype)
                point_group_ids.view(-1)[(labels >=0).view(-1)] = token_point_group_id.view(-1)[is_patch_token.view(-1)]

                # Tokens cannot predict patches before a patch previous points in the group
                # have predicted
                starts_point_group = point_group_ids[:, :-1] != point_group_ids[:, 1:]
                less_then = (labels[:, :-1] * (~starts_point_group).float())
                candidates = torch.arange(patch_logits.shape[-1], device=labels.device)
                mask = candidates[None, None, :] >= less_then[:, :, None]
                next_token_mask = torch.nn.functional.pad(mask, [0, 0, 1, 0], value=True)

                if subsegment_ids is not None:
                    # Mask out patches in images from a previous packed inputs
                    patch_subegments = torch.zeros([batch_size, labels.shape[1]], device=labels.device, dtype=subsegment_ids.dtype)
                    patch_subegments.view(-1)[(labels >=0).view(-1)] = subsegment_ids.view(-1)[is_patch_token.view(-1)]

                    image_subsegment = torch.zeros([batch_size, image_features_mask.shape[1]], device=labels.device, dtype=subsegment_ids.dtype)
                    image_subsegment.view(-1)[image_features_mask.view(-1)] = subsegment_ids.view(-1)[is_image_token.view(-1)]

                    patch_example = patch_subegments // EXAMPLE_SUBSEGMENT_INCREMENT
                    image_example = image_subsegment // EXAMPLE_SUBSEGMENT_INCREMENT
                    same_segment = (patch_example[:, :, None] == image_example[:, None, :])
                    if self.config.no_more_points_class:
                        same_segment = F.pad(same_segment, [0, 1], value=True)
                    next_token_mask = next_token_mask & same_segment
                patch_logits = torch.where(next_token_mask, patch_logits, -100000000)
            else:
                if subsegment_ids is not None:
                    raise NotImplementedError()

            if self.config.no_more_points_class:
                labels = torch.where(labels == NO_POINTS_LABEL, patch_k.shape[1]-1, labels)
            patch_loss = _weighted_cross_entropy(
                patch_logits, labels, loss_masks, patch_predictor_tokens, labels_padded=True)
            patch_acc = (torch.argmax(patch_logits, -1) == labels).float().sum()
            num_logits = (labels >= 0).float().sum()
            metrics["patch_loss"] = (patch_loss, num_logits)
            metrics["patch_acc"] = (patch_acc, num_logits)
            token_losses.append(patch_loss)
        else:
            # Dummy forward pass so FSDP models stay in sync
            image_q = self.patch_q(x[:, :0])

        # Make subpatch predictions
        if doing_inference:
            if target_patch_ids is not None:
                assert seq_len == 1
                assert target_patch_ids.shape[1] == 1
                subpatch_point_q = self.subpatch_q(x_norm.squeeze(1))
                subpatch_k = subpatch_k[batch_idx, with_subpatch_patch_ids.squeeze(1)]
                subpatch_logits = torch.einsum("pd,pcd->pc", subpatch_point_q, subpatch_k)
                if self.config.norm_logits:
                    subpatch_logits = subpatch_logits / math.sqrt(patch_k.shape[-1])
                if self.config.mask_subpatches != MaskTokens.never:
                    subpatch_mask = vit_features_mask[batch_idx, with_subpatch_patch_ids.squeeze(1)]
                    subpatch_logits = torch.where(subpatch_mask, subpatch_logits, -100000)
            else:
                subpatch_point_q = None
        elif any_subpatch_predictors and not doing_prefilling:
            subpatch_x = x_norm.view(-1, x.shape[-1])[subpatch_predictor_tokens.view(-1)]
            subpatch_point_q = self.subpatch_q(subpatch_x)
            subpatch_k = subpatch_k[batch_idx[:, None], torch.clamp(with_subpatch_patch_ids, min=0)]
            subpatch_k = subpatch_k.view(-1, vit_features.shape[-2], subpatch_k.shape[-1])[target_subpatch_ids.view(-1) >= 0]
            subpatch_logits = torch.einsum("pd,pcd->pc", subpatch_point_q, subpatch_k)
            if self.config.norm_logits:
                subpatch_logits = subpatch_logits / math.sqrt(patch_k.shape[-1])
            if self.config.mask_subpatches == MaskTokens.always:
                subpatch_mask = vit_features_mask[batch_idx[:, None], with_subpatch_patch_ids]
                subpatch_mask = subpatch_mask.view(-1, subpatch_mask.shape[-1])[(target_subpatch_ids >= 0).view(-1)]
                subpatch_logits = torch.where(subpatch_mask, subpatch_logits, -100000)
            if target_patch_ids is not None:
                vit_targets = target_subpatch_ids.view(-1)[target_subpatch_ids.view(-1) >= 0]
                subpatch_loss = _weighted_cross_entropy(
                    subpatch_logits, vit_targets, loss_masks, subpatch_predictor_tokens)
                subpatch_acc = (torch.argmax(subpatch_logits, dim=-1) == vit_targets).float().sum()
                num_points = subpatch_k.shape[0]
                metrics["subpatch_loss"] = (subpatch_loss, num_points)
                metrics["subpatch_acc"] = (subpatch_acc, num_points)
                token_losses.append(subpatch_loss)
        else:
            # Dummy forward pass so FSDP models stay in sync
            subpatch_point_q = self.subpatch_q(x[:, :0])

        # Make patch-location predictions
        if self.subpatch_loc_k is not None:
            is_loc_token = input_ids == self._target_subpatch_location
            if doing_inference:
                if point_target_ids is not None:
                    location_logits = self.subpatch_loc_k(x.squeeze(1))
                else:
                    location_logits = None
            elif torch.any(is_loc_token):
                loc_predictor_tokens = F.pad(is_loc_token[:, 1:], [0, 1])
                loc_x = x_norm.view(-1, x.shape[-1])[loc_predictor_tokens.view(-1)]
                location_logits = self.subpatch_loc_k(loc_x)
                if point_target_ids is not None:
                    location_ids = point_target_ids[:, :, 2]
                    location_targets = location_ids.reshape(-1)[location_ids.reshape(-1) >= 0]
                    location_loss = _weighted_cross_entropy(
                        location_logits, location_targets, loss_masks, loc_predictor_tokens)
                    location_acc = (torch.argmax(location_logits, dim=-1) == location_targets).float().sum()
                    num_points = location_targets.shape[0]
                    metrics["location_loss"] = (location_loss, num_points)
                    metrics["location_acc"] = (location_acc, num_points)
                    token_losses.append(location_loss)
            else:
                # Dummy forward pass so FSDP models stay in sync
                location_logits = self.subpatch_loc_k(x[:, :0])

        # Add metrics and dummy losses so if the batch didn't have any points
        # FSDP models stay in sync even on the backwards pass
        if not doing_inference and not doing_prefilling:
            required_keys = [
                ("patch", lambda: (patch_k.sum() + image_q.sum())*0),
                ("subpatch", lambda: (subpatch_point_q.sum() + subpatch_k.sum())*0),
            ]
            if self.subpatch_loc_k is not None:
                required_keys.append(("location", lambda: location_logits.sum()*0))
            o = torch.ones((), dtype=x.dtype, device=dev)
            for key, value_gen in required_keys:
                if f"{key}_loss" not in metrics:
                    p = value_gen()
                    metrics[f"{key}_loss"] = (p, o)
                    metrics[f"{key}_acc"] = (p, o)
                    # We need a fake loss so the backward pass also stays in sync
                    token_losses.append(p)
            token_loss = token_losses[0]
            for l in token_losses[1:]:
                token_loss = token_loss + l
            metrics["token_losses"] = token_loss

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)
        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        if response_logits_only:
            # Get regular logits
            logits_x = x.view(-1, x.shape[-1])[response_mask.view(-1)]
            if self.config.llm.weight_tying:
                logits = self.transformer.wte(logits_x, logits_with_new_embedding=True)
            else:
                logits = self.transformer.ff_out(logits_x)  # type: ignore
            if self.config.llm.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.llm.d_model))
        else:
            if self.config.llm.weight_tying:
                logits = self.transformer.wte(x, logits_with_new_embedding=True)
            else:
                logits = self.transformer.ff_out(x)  # type: ignore
            if self.config.llm.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.llm.d_model))

        if doing_prefilling:
            image_data = ImageCache(
                patch_k=patch_k,
                subpatch_k=subpatch_k,
                vit_features=vit_features,
                patch_k_mask=patch_k_mask,
                token_pooling=token_pooling,
                image_pos_ids=image_pos_ids,
                image_features0=image_features,
                total_image_tokens=num_image_tokens,
            )

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            metrics=metrics,
            patch_logits=patch_logits,
            subpatch_logits=subpatch_logits,
            location_logits=location_logits,
            image_data_cache=image_data
        )

    def generate(
        self,
        batch,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        is_distributed: bool=False,
        force_single_point: bool=False
    ) -> OLMoGenerateOutput:
        if beam_size != 1:
            raise NotImplementedError()
        input_ids: torch.LongTensor = batch["input_ids"]
        attention_mask: Optional[torch.Tensor] = batch.get("attention_mask")
        image_args = dict(
            images=batch.get("images"),
            image_masks=batch.get("image_masks"),
            token_pooling=batch.get("token_pooling"),
            image_pos_ids=batch.get("image_pos_ids"),
        )
        image_pooling = None
        batch_size = input_ids.shape[0]
        token_targets = [[] for _ in range(batch_size)]
        vit_targets = [[] for _ in range(batch_size)]
        location_targets = [[] for _ in range(batch_size)]
        _end_of_points_token_id = self.config.build_tokenizer().encode("\">")[0]

        llm_cfg = self.config.llm

        beam_search = BeamSearch(
            llm_cfg.build_tokenizer().eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
            distributed_model=is_distributed
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        mask_len = seq_len + max_steps if llm_cfg.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        image_data = None
        if llm_cfg.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_steps))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= llm_cfg.max_sequence_length
            )

        tokens_generated = 0
        prev_target_ids = None
        max_vit_val = None

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.llm.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            nonlocal position_ids
            nonlocal image_args
            nonlocal image_data
            nonlocal max_vit_val
            nonlocal append_last_valid_logits
            nonlocal prev_target_ids
            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if not llm_cfg.use_position_ids and attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
                if llm_cfg.use_position_ids:
                    position_ids = position_ids[:, -1:] + 1
                    _, *last_dims = position_ids.size()
                    _position_ids = (
                        position_ids.unsqueeze(1)
                        .expand(batch_size, beam_size, *last_dims)
                        .reshape(batch_size * beam_size, *last_dims)
                    )
                else:
                    _position_ids = None

                # Will store the point we are generating, or just finished, or be -1 if there is no
                # no such point
                if self.config.patch_location:
                    point_target_ids = torch.zeros([batch_size, 1, 3], device=self.device, dtype=torch.long) - 1
                else:
                    point_target_ids = torch.zeros([batch_size, 1, 2], device=self.device, dtype=torch.long) - 1

                # Collect predicted points
                end_of_points = False
                any_points = False
                for b_id, input_id in enumerate(input_ids):
                    b_tokens = token_targets[b_id]
                    b_vit = vit_targets[b_id]
                    if input_id == self._target_patch_token:
                        if (
                            self.config.mask_repeats != MaskTokens.never and
                            len(b_tokens) > 0 and
                            len(b_vit) > 0 and
                            b_vit[-1] == max_vit_val
                        ):
                            # If we predicted the same patch as before, will have to repeat
                            state["patch_logits"][b_id, :, :b_tokens[-1]+1] = -100000
                        if self.config.mask_patches != MaskTokens.never and len(b_tokens) > 0:
                            state["patch_logits"][b_id, :, :b_tokens[-1]] = -100000
                        if self.config.no_more_points_class and len(b_tokens) == 0:
                            # Our format require generating at least one point
                            state["patch_logits"][b_id, :, -1] = -100000

                        ix = torch.argmax(state["patch_logits"][b_id], dim=-1)[0]
                        if self.config.no_more_points_class and ix == (state["patch_logits"][b_id].shape[-1]) -1:
                            ix = NO_POINTS_LABEL
                        b_tokens.append(ix)
                        point_target_ids[b_id, 0, 0] = ix
                        any_points = True
                    if input_id == self._target_subpatch_token:
                        if (
                            len(b_tokens) > 1 and
                            len(b_vit) > 0 and
                            b_tokens[-2] == b_tokens[-1]
                        ):
                            if self.config.mask_repeats != MaskTokens.never:
                                state["subpatch_logits"][b_id, :b_vit[-1]+1] = -100000
                            elif self.config.mask_patches != MaskTokens.never:
                                state["subpatch_logits"][b_id, :b_vit[-1]] = -100000

                        ix = torch.argmax(state["subpatch_logits"][b_id], dim=-1)
                        vit_targets[b_id].append(ix)
                        if len(token_targets[b_id]) > 0:
                            point_target_ids[b_id, 0, 0] = token_targets[b_id][-1]
                        else:
                            point_target_ids[b_id, 0, 0] = 0
                        point_target_ids[b_id, 0, 1] = ix
                        any_points = True
                        max_vit_val = (state["subpatch_logits"].shape[-1]-1)
                    if input_id == self._target_subpatch_location:
                        ix = torch.argmax(state["location_logits"][b_id], dim=-1)
                        location_targets[b_id].append(ix)
                        point_target_ids[b_id, 0, 0] = token_targets[b_id][-1]
                        point_target_ids[b_id, 0, 1] = vit_targets[b_id][-1]
                        point_target_ids[b_id, 0, 2] = ix
                        any_points = True
                if any_points:
                    prev_target_ids = point_target_ids
                _image_args = {}
                _append_last_valid_logits = None
                _point_target_ids = point_target_ids if any_points else None
            else:
                point_target_ids = None
                past_key_values = None
                input_ids = state["input_ids"]
                _point_target_ids = None
                _image_args = image_args
                _position_ids = position_ids
                _append_last_valid_logits = append_last_valid_logits

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
                point_target_ids=_point_target_ids,
                append_last_valid_logits=_append_last_valid_logits,
                image_data=image_data,
                prev_target_ids=prev_target_ids,
                **_image_args
            )

            if force_single_point and (tokens_generated-1) < len(self._single_point_seq):
                output.logits[:, :, self._single_point_seq[tokens_generated-1]] = 1000000
                if tokens_generated == 9:
                    output.patch_logits[:, :, -1] = 100000

            # Force the model to only generate (patch, subpatch, location) triples, and to stop
            # pointing after a `NO_POINTS_LABEL` prediction
            if prev_target_ids is None:
                output.logits[:, :, self._target_subpatch_location] = -1000000
                output.logits[:, :, self._target_subpatch_token] = -1000000
                if tokens_generated == 1:
                    # We don't compute logits during pre-filling so we shouldn't start a point
                    output.logits[:, :, self._target_patch_token] = -1000000
            else:
                for b_ix, prev_point in enumerate(prev_target_ids[:, 0]):
                    if (
                        self.config.no_more_points_class and
                        len(token_targets[b_ix]) > 0 and
                        token_targets[b_ix][-1] == NO_POINTS_LABEL
                    ):
                        # predicted no-more-points, not more pointing labels allowed
                        output.logits[b_ix, :, self._target_patch_token] = -1000000
                        output.logits[b_ix, :, self._target_subpatch_location] = -1000000
                        output.logits[b_ix, :, self._target_subpatch_token] = -1000000
                    elif prev_point[0] == -1:
                        # Can't generate subpatch/location until we predict patch
                        output.logits[b_ix, :, self._target_subpatch_location] = -1000000
                        output.logits[b_ix, :, self._target_subpatch_token] = -1000000
                    elif prev_point[1] == -1:
                        # Predicted a patch, must predict subpatch
                        output.logits[b_ix, :, self._target_subpatch_token] = 1000000
                    elif len(prev_point) == 3 and prev_point[2] == -1:
                        # Predicted a subpatch, must predict location
                        output.logits[b_ix, :, self._target_subpatch_location] = 1000000
                    else:
                        # Can't generate subpatch/location until we predict patch
                        output.logits[b_ix, :, self._target_subpatch_location] = -1000000
                        output.logits[b_ix, :, self._target_subpatch_token] = -1000000

            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)
            if tokens_generated == 1:
                # Cache image data in case we need compute token logits
                image_data = output.image_data_cache

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias
            state["patch_logits"] = output.patch_logits
            state["subpatch_logits"] = output.subpatch_logits
            state["location_logits"] = output.location_logits
            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            token_ids, scores = beam_search.search(initial_preds, state, step)

        token_target_ids = []
        for b_id in range(len(input_ids)):
            if self.config.patch_location:
                token_target_ids.append(torch.tensor(list(
                    zip(token_targets[b_id], vit_targets[b_id], location_targets[b_id]))))
            else:
                token_target_ids.append(torch.tensor(list(zip(token_targets[b_id], vit_targets[b_id]))))

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
            token_target_ids=token_target_ids,
        )
