from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import init

from olmo.nn.llm import RopeType, RMSLayerNorm
from olmo.torch_util import BufferCache


class PadWithLearnedVector(nn.Module):
    """Module that pads vector

    Used to add in the no-more-point key value
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.vector = nn.Parameter(torch.zeros([dim]))

    def reset_parameters(self):
        init.zeros_(self.vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vector = torch.tile(self.vector[None, :], [x.shape[0], 1])
        return torch.concatenate([x, vector[:, None, :]], dim=1)


class FlatRotaryEmbedding(nn.Module):
    """Rotary embedding module that operates on an individual [batch, seq_len] array

    This is used for rotating patch key/queries in MolmoPoint
    """

    def __init__(self, rope_theta, dim, cache_prefix, cache: BufferCache, max_len=None, device=None):
        super().__init__()
        self.dim = dim
        self.cache_prefix = cache_prefix
        self.rope_theta = rope_theta
        self.max_len = max_len
        self.__cache = cache

    def warmup_cache(self, device, cp_enabled: bool = False):
        if self.max_len is not None:
            if isinstance(self.dim, (tuple, list)):
                for dim in self.dim:
                    self.get_rotary_embedding(dim, self.max_len, device)
            else:
                self.get_rotary_embedding(self.dim, self.max_len, device)

    def compute_inv_frequency(
        self,
        seq_len: int,
        dim: int,
        device: torch.device,
        rope_type: RopeType,
    ) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        )

    def get_rotary_embedding(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        rope_type: Optional[RopeType] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sin_key = f"{self.cache_prefix}rope_pos_sin{dim}"
        cos_key = f"{self.cache_prefix}rope_pos_cos{dim}"
        if (
            (pos_sin := self.__cache.get(sin_key)) is not None
            and (pos_cos := self.__cache.get(cos_key)) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                # This guard seems to prevent certain kinds of compiling errors and graphs breaks,
                # presumably due to the buffer cache modification confusing the compiler,
                # but it is hard to pin down why its sometimes needed and sometimes isn't
                if not torch.compiler.is_compiling():
                    self.__cache[sin_key] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                if not torch.compiler.is_compiling():
                    self.__cache[cos_key] = pos_cos
            return pos_sin[:seq_len, :], pos_cos[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = self.compute_inv_frequency(seq_len, dim, device, rope_type)
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin(), positions.cos()
        self.__cache[sin_key] = pos_sin
        self.__cache[cos_key] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, hs = x.size()
        x = x.view(B, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        pos_ids: torch.Tensor,
        max_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_ = q.float()

        with torch.autocast(q.device.type, enabled=False):
            seq_len = q_.shape[0]
            if isinstance(self.dim, int):
                pos_ids = pos_ids.squeeze(-1)
                pos_sin, pos_cos = self.get_rotary_embedding(self.dim, max_len, q_.device)
                pos_sin = pos_sin.type_as(q_)[pos_ids]
                pos_cos = pos_cos.type_as(q_)[pos_ids]
                q_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, q_,)
            else:
                parts = []
                for q_part, dim, pos_ids_part in zip(torch.split(q, self.dim, -1), self.dim, torch.unbind(pos_ids, -1)):
                    pos_sin, pos_cos = self.get_rotary_embedding(dim, max_len, q_.device)
                    pos_sin = pos_sin.type_as(q_)[pos_ids_part]
                    pos_cos = pos_cos.type_as(q_)[pos_ids_part]
                    parts.append(self.apply_rotary_pos_emb(pos_sin, pos_cos, q_part))
                q_ = torch.cat(parts, dim=-1)
        return q_.type_as(q)
