import dataclasses
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from olmo.config import BaseConfig
from olmo.models.molmo_point.molmo_point_example_preprocessor import NO_POINTS_LABEL
from olmo.preprocessing.preprocessor_utils import TOKEN_POOLING_KEYS
from olmo.dist_util import get_cp_mesh
from olmo.torch_util import get_rank


EXAMPLE_SUBSEGMENT_INCREMENT = 100000


def compute_image_shard_boundaries(
    cum_image_bounds: np.ndarray,
    cum_token_pooling_bounds: np.ndarray,
    cp_world_size: int
) -> Dict[int, Dict[str, int]]:
    """
    Pre-compute image shard boundaries for each GPU rank.
    
    Args:
        cum_image_bounds: Cumulative image counts for each input
        cum_token_pooling_bounds: Cumulative token pooling row counts
        cp_world_size: Number of GPUs in context parallel group
        
    Returns:
        Dictionary mapping GPU rank to {start_image_idx, end_image_idx, start_pool_idx, end_pool_idx, max_num_tokens}
    """
    if len(cum_image_bounds) == 0:
        return {}
    
    total_images = cum_image_bounds[-1]
    images_per_gpu = total_images / cp_world_size
    
    current_idx = 0
    max_num_tokens = 0
    boundaries = {}
    
    for gpu_rank in range(cp_world_size):
        gpu_start_idx = current_idx
        target_images_for_this_gpu = (gpu_rank + 1) * images_per_gpu
        
        # Find where to end for this GPU
        gpu_end_idx = current_idx
        for i in range(current_idx, len(cum_image_bounds)):
            images_if_we_take_this = cum_image_bounds[i]
            if images_if_we_take_this >= target_images_for_this_gpu:
                gpu_end_idx = i + 1
                break
        else:
            gpu_end_idx = len(cum_image_bounds)
        
        # Calculate actual indices
        start_image_idx = cum_image_bounds[gpu_start_idx - 1] if gpu_start_idx > 0 else 0
        end_image_idx = cum_image_bounds[gpu_end_idx - 1] if gpu_end_idx > 0 else 0
        
        start_pool_idx = cum_token_pooling_bounds[gpu_start_idx - 1] if gpu_start_idx > 0 else 0
        end_pool_idx = cum_token_pooling_bounds[gpu_end_idx - 1] if gpu_end_idx > 0 else 0
        
        gpu_num_tokens = end_pool_idx - start_pool_idx
        max_num_tokens = max(max_num_tokens, gpu_num_tokens)
        
        boundaries[gpu_rank] = {
            'start_image_idx': int(start_image_idx),
            'end_image_idx': int(end_image_idx),
            'start_pool_idx': int(start_pool_idx),
            'end_pool_idx': int(end_pool_idx),
        }
        
        current_idx = gpu_end_idx
    
    # Post-process to ensure every GPU has at least some work
    # We may need multiple passes since stealing from a GPU might leave it with zero images
    max_iterations = cp_world_size  # Upper bound on iterations needed
    for iteration in range(max_iterations):
        any_changes = False
        
        for gpu_rank in range(cp_world_size):
            if boundaries[gpu_rank]['start_image_idx'] == boundaries[gpu_rank]['end_image_idx']:
                # This GPU has no work, try to steal from a neighbor
                stolen = False
                
                # First try stealing from the next GPU (forward direction)
                if gpu_rank < cp_world_size - 1:
                    next_gpu = boundaries[gpu_rank + 1]
                    # Find the input boundary indices for the next GPU
                    next_start_idx = None
                    next_second_idx = None
                    for i in range(len(cum_image_bounds)):
                        if cum_image_bounds[i] == next_gpu['start_image_idx']:
                            next_start_idx = i
                        elif cum_image_bounds[i] > next_gpu['start_image_idx'] and next_second_idx is None:
                            next_second_idx = i
                            break
                    
                    if next_start_idx is not None and next_second_idx is not None:
                        # Next GPU has at least 2 inputs, steal the first one
                        new_next_start_image = cum_image_bounds[next_second_idx]
                        new_next_start_pool = cum_token_pooling_bounds[next_second_idx]
                        
                        boundaries[gpu_rank]['start_image_idx'] = next_gpu['start_image_idx']
                        boundaries[gpu_rank]['end_image_idx'] = int(new_next_start_image)
                        boundaries[gpu_rank]['start_pool_idx'] = next_gpu['start_pool_idx']
                        boundaries[gpu_rank]['end_pool_idx'] = int(new_next_start_pool)
                        
                        boundaries[gpu_rank + 1]['start_image_idx'] = int(new_next_start_image)
                        boundaries[gpu_rank + 1]['start_pool_idx'] = int(new_next_start_pool)
                        
                        stolen = True
                        any_changes = True
                
                # If couldn't steal from next, try stealing from previous GPU
                if not stolen:
                    donor_rank = gpu_rank - 1
                    while donor_rank >= 0 and not stolen:
                        donor = boundaries[donor_rank]
                        # Find the input boundary indices for the donor
                        donor_start_idx = None
                        donor_second_last_idx = None
                        donor_end_idx = None
                        
                        for i in range(len(cum_image_bounds)):
                            if cum_image_bounds[i] == donor['start_image_idx']:
                                donor_start_idx = i
                            if cum_image_bounds[i] == donor['end_image_idx']:
                                donor_end_idx = i
                            if cum_image_bounds[i] < donor['end_image_idx']:
                                donor_second_last_idx = i
                        
                        if (donor_start_idx is not None and donor_end_idx is not None and 
                            donor_second_last_idx is not None and donor_second_last_idx > donor_start_idx):
                            # Donor has at least 2 inputs, steal the last one
                            new_donor_end_image = cum_image_bounds[donor_second_last_idx]
                            new_donor_end_pool = cum_token_pooling_bounds[donor_second_last_idx]
                            
                            boundaries[gpu_rank]['start_image_idx'] = int(new_donor_end_image)
                            boundaries[gpu_rank]['end_image_idx'] = donor['end_image_idx']
                            boundaries[gpu_rank]['start_pool_idx'] = int(new_donor_end_pool)
                            boundaries[gpu_rank]['end_pool_idx'] = donor['end_pool_idx']
                            
                            boundaries[donor_rank]['end_image_idx'] = int(new_donor_end_image)
                            boundaries[donor_rank]['end_pool_idx'] = int(new_donor_end_pool)
                            
                            stolen = True
                            any_changes = True
                            
                        donor_rank -= 1
        
        # If no changes were made in this iteration, we're done
        if not any_changes:
            break
    
    # Recalculate max_num_tokens after all redistributions
    max_num_tokens = 0
    for gpu_rank in range(cp_world_size):
        gpu_tokens = boundaries[gpu_rank]['end_pool_idx'] - boundaries[gpu_rank]['start_pool_idx']
        max_num_tokens = max(max_num_tokens, gpu_tokens)
    
    # Add max_num_tokens to all boundaries
    for gpu_rank in boundaries:
        boundaries[gpu_rank]['max_num_tokens'] = int(max_num_tokens)
    
    return boundaries


def pack(*examples: Dict) -> Dict:
    keys = set()
    for ex in examples:
        keys.update(ex)
    keys = {k for k in keys if "metadata" != k}
    if "subsegment_ids" not in keys:
        keys.add("subsegment_ids")
    patch_keys = [
        k for k in TOKEN_POOLING_KEYS if k in keys]
    if "images" in keys:
        assert len(patch_keys) > 0, "Example had images but no image->token mapping idx"
    image_offset = 0

    cum_token_pooling_bounds = []
    cum_image_bounds = []

    for example_ix, example in enumerate(examples):
        # Patch indices need to be offset by total number of images patches
        if "images" in example:
            n_patches = np.prod(example["images"].shape[:2])
            for k in patch_keys:
                if k in example:
                    assert np.all(example[k] < n_patches)
                    example[k] = np.where(example[k] >= 0, example[k] + image_offset, example[k])
            if 'cum_token_pooling_bounds' in example:
                bounds = example['cum_token_pooling_bounds']
            else:
                bounds = [example['token_pooling'].shape[0]]

            if len(cum_token_pooling_bounds) == 0:
                cum_token_pooling_bounds = list(bounds)
            else:
                last_bound = cum_token_pooling_bounds[-1]
                cum_token_pooling_bounds.extend([b + last_bound for b in bounds])

            if 'cum_image_bounds' in example:
                img_bounds = example['cum_image_bounds']
            else:
                img_bounds = [example['images'].shape[0]]

            if len(cum_image_bounds) == 0:
                cum_image_bounds = list(img_bounds)
            else:
                last_img_bound = cum_image_bounds[-1]
                cum_image_bounds.extend([b + last_img_bound for b in img_bounds])
            
            example["image_offset"] = image_offset

            image_offset += n_patches
            assert "position_ids" in example
        n_tokens = len(example["input_tokens"])

        # Modify or add subsegment ids to prevent intra-example attention
        # Tokens can only attend to subsegments >= then their subsegment, so
        # we give example increasing subsegments to prevent cross-example attention
        example_subsegemnt_id = example_ix*EXAMPLE_SUBSEGMENT_INCREMENT
        if "subsegment_ids" not in example:
            example["subsegment_ids"] = np.full([n_tokens], example_subsegemnt_id)
        else:
            example["subsegment_ids"] += example_subsegemnt_id

    if "images" in keys:
        img = next(iter(ex for ex in examples if "images" in ex))["images"]
        for ex in examples:
            if "images" not in ex:
                ex["images"] = np.zeros([0]+list(img.shape[1:]), dtype=img.dtype)

    offset = 0
    if "point_target_ids" in keys:
        dim = [example["point_target_ids"] for example in examples
               if "point_target_ids" in example][0].shape[1]
        for ex in examples:
            if "point_target_ids" in ex:
                target_ids = ex["point_target_ids"]
                patch_ids = target_ids[:, 0]
                target_ids[:, 0] = np.where(
                    (patch_ids >= 0) & (patch_ids != NO_POINTS_LABEL), patch_ids + offset, patch_ids)
            else:
                ex["point_target_ids"] = np.zeros([0, dim], dtype=np.int64)
            if "token_pooling" in ex:
                n_image_tokens = np.any(ex["token_pooling"] >= 0, -1).sum()
                offset += n_image_tokens

    for ex in examples:
        for key in patch_keys:
            max_pooling_shape = max(ex[key].shape[1] for ex in examples if key in ex)
            if key not in ex:
                ex[key] = np.full([0, max_pooling_shape], -1)
            elif ex[key].shape[1] < max_pooling_shape:
                delta = max_pooling_shape - ex[key].shape[1]
                ex[key] = np.pad(ex[key], [[0, 0], [0, delta]], constant_values=-1)
    out = {k: np.concatenate([ex.get(k, np.array([], dtype=np.int64 if 'bounds' in k else np.float32)) \
                              for ex in examples], axis=0) for k in keys if k != "metadata"}
    out["cum_token_pooling_bounds"] = np.array(cum_token_pooling_bounds, dtype=np.int64)
    out["cum_image_bounds"] = np.array(cum_image_bounds, dtype=np.int64)
    out["offset"] = np.array([ex.get("image_offset", -1) for ex in examples], dtype=np.int64)
    out["metadata"] = [ex.get("metadata", {}) for ex in examples]
    return out


def pack_with_cp(*examples: Dict, cp_world_size: int = 1) -> Dict:
    """Pack examples and compute image shard boundaries for context parallelism."""
    out = pack(*examples)
    
    # Pre-compute image shard boundaries if using context parallelism
    if cp_world_size > 1 and len(out.get("cum_image_bounds", [])) > 0:
        cum_image_bounds = out["cum_image_bounds"]
        cum_token_pooling_bounds = out["cum_token_pooling_bounds"]
        
        # Compute boundaries for each GPU rank
        boundaries = compute_image_shard_boundaries(
            cum_image_bounds,
            cum_token_pooling_bounds,
            cp_world_size
        )
        # Store as numpy arrays for each field
        # Create arrays indexed by GPU rank
        if boundaries:
            if 'offset' in out:
                boundaries['offset'] = out['offset']
                boundaries['metadata'] = out['metadata']

            out["image_shard_boundaries"] = boundaries
    
    return out


def packed_iterator(it, packer):
    for i, ex in enumerate(it):
        out = packer(i, ex)
        if out is not None:
            yield out


@dataclasses.dataclass
class PackingConfig(BaseConfig):
    buffer_size: int = 32
    mode: str = "dynamic_solver"

    text_weight: float = 1.0
    """Text token weight for the dynamic solver"""

    image_weight: float = 1.0
    """Image token weight for the dynamic solver"""

    shortcut_max_len_images: bool = False
    """Don't buffer examples that have the max number of images"""

    track_packing_state: bool = True

    cp_world_size: int = 1
    """Context parallel world size for pre-computing image shard boundaries"""

    def bulid(self, text_max_len, image_max_len, mesh=None):
        if self.mode == "dynamic_solver":
            text_c = Constraint("input_tokens", text_max_len, True, self.text_weight, max(1, text_max_len//512))
            image_c = Constraint("images", image_max_len, self.shortcut_max_len_images, self.image_weight, 1)
            return DynamicSolver(self.buffer_size, [text_c, image_c], mesh=mesh, cp_world_size=self.cp_world_size)
        else:
            raise ValueError(self.mode)


@dataclasses.dataclass
class Constraint:
    key: str
    """Key in example dictionaries to contain"""

    max_len: int
    """Max total length of `key` tensors in the packed examples"""

    allow_shortcut: bool
    """Don't buffer examples that are at `max_len` on their own"""

    weight: float
    """Value to put on this example in the solver"""

    granularity: int
    """Granularity to run the solver at"""

    def get_quantized_value(self, val: int):
        return (val + self.granularity - 1) // self.granularity

    def get_quantized_max_len(self):
        return self.get_quantized_value(self.max_len)


def select_subset_2d_knapsack(t_values, i_values, max_t, max_i, obj_vals):
    """Vectorized 2D knapsack dynamic program solver"""
    M = len(t_values)

    # DP table with quantized dimensions
    dp = np.zeros((M + 1, max_t + 1, max_i + 1), dtype=np.float32)

    # Vectorized DP fill
    for item in range(1, M + 1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        obj_val = obj_vals[item - 1]

        # Copy previous layer
        dp[item] = dp[item - 1]

        # Vectorized update where item can fit
        if t_val_q <= max_t and i_val_q <= max_i:
            # Create shifted view for the "take item" case
            take_val = dp[item - 1, :max_t + 1 - t_val_q, :max_i + 1 - i_val_q] + obj_val

            # Update positions where taking item is better
            dp[item, t_val_q:, i_val_q:] = np.maximum(
                dp[item, t_val_q:, i_val_q:],
                take_val
            )

    # Backtrack to find solution
    selected_indices = []
    t_rem_q, i_rem_q = max_t, max_i

    for item in range(M, 0, -1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]

        if (t_val_q <= t_rem_q and i_val_q <= i_rem_q and
            dp[item, t_rem_q, i_rem_q] != dp[item - 1, t_rem_q, i_rem_q]):
            selected_indices.append(item - 1)
            t_rem_q -= t_val_q
            i_rem_q -= i_val_q
    return selected_indices


@dataclasses.dataclass
class BufferedExample:
    example_id: int
    value: float
    quantized_lens: Dict[str, int]
    example: Dict


class DynamicSolver:
    """Pack examples by running a dynamic program to optimize what examples to pack"""

    def __init__(self, max_buffer_size: int, constraints: List[Constraint], verbosity=0,
                 cp_world_size: int = 1, mesh=None):
        self.max_buffer_size = max_buffer_size
        self._buffer: List[BufferedExample] = []
        self.verbosity = verbosity
        self.constraints = constraints
        if mesh is not None:
            cp_pg = get_cp_mesh(mesh).get_group()
            self.cp_rank = get_rank(cp_pg)
        else:
            self.cp_rank = 0
        self.cp_world_size = cp_world_size

    def _example_str(self, example):
        return ' '.join(f"{c.key}={len(example.get(c.key, []))}" for c in self.constraints)

    def get_buffered_example_ids(self):
        return [x.example_id for x in self._buffer]

    def __call__(self, example_id: int, example: Dict) -> List:
        for constraint in self.constraints:
            if constraint.allow_shortcut:
                m = len(example[constraint.key])
                if constraint.get_quantized_value(m) >= constraint.get_quantized_max_len():
                    if self.verbosity > 1:
                        print(f"Example already at constraints: {self._example_str(example)}")
                    return pack_with_cp(example, cp_world_size=self.cp_world_size)

        quantized_lens = {}
        value = 0
        for c in self.constraints:
            if c.key in example:
                quantized_lens[c.key] = c.get_quantized_value(len(example[c.key]))
                value += len(example[c.key])*c.weight
            else:
                quantized_lens[c.key] = 0
        buffered_example = BufferedExample(
            example_id, quantized_lens=quantized_lens, value=value, example=example)

        if len(self._buffer) < self.max_buffer_size:
            self._buffer.append(buffered_example)
            if self.verbosity > 1:
                print(f"Add to pool: {self._example_str(example)}, buffer_sz{len(self._buffer)}")
            return None

        if len(self.constraints) != 2:
            raise NotImplementedError("Solver currently only supports exactly 2 constraints")

        if self.verbosity > 1:
            for c in self.constraints:
                print(f"{c.key}: {[len(x.example.get(c.key, [])) for x in self._buffer]}")

        c1, c2 = list(self.constraints)
        indices = select_subset_2d_knapsack(
            [ex.quantized_lens[c1.key] for ex in self._buffer],
            [ex.quantized_lens[c2.key] for ex in self._buffer],
            c1.get_quantized_max_len(),
            c2.get_quantized_max_len(),
            [ex.value for ex in self._buffer]
        )
        if len(indices) == 0:
            raise RuntimeError("No indices returned by select_subset_2d_knapsack")

        if self.verbosity > 0:
            print(f"Yield {indices}")
            for c in self.constraints:
                vals = [len(self._buffer[i].example.get(c.key, ())) for i in indices]
                print(f"{c.key}: {sum(vals)} {vals}")

        out = pack_with_cp(*(self._buffer[i].example for i in indices), cp_world_size=self.cp_world_size)
        for ix in sorted(indices, reverse=True):
            self._buffer.pop(ix)
        self._buffer.append(buffered_example)
        return out


