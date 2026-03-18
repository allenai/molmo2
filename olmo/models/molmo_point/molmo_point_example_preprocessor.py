import dataclasses
import logging
import re
from collections import defaultdict
from enum import IntEnum
from typing import Any, Optional, Dict, Callable, List, Union, Tuple
import numpy as np

from olmo.config import D
from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter, Message
from olmo.models.molmo_point.molmo_point_text_preprocessor import \
    MolmoPointInterleavedTextPreprocessor, MolmoPointTextPreprocessorConfig
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig, MultiCropImagePreprocessor, \
    MultiImagePreprocessor
from olmo.tokenizer import TOKEN_INDEX_TOKEN, LOCATION_CLS_TOKEN
from olmo.util import flatten_lists

from olmo.preprocessing.image_preprocessor import load_image
from olmo.preprocessing.text_preprocessor import InterleavedTextPreprocessor
from olmo.data.video_loader import VideoFrames
from olmo.preprocessing.preprocessor_utils import TensorSpec, VariablePaddingSpec, \
    TokenizedVisionData
import math


log = logging.getLogger(__name__)


def get_patch_coordinates(point, patch_idx_arr, pooling_idx, location=None):
    """Build ground truth target ids for a point

    point: (x, y) normalized coordinates between 0 and 1
    patch_idx_arr: Mapping of patch x,y coordinates -> patch_id
    pooling_idx: Mapping of token_id -> patch_ids pooled for that token
    location: How/whether to include location targets
    """
    x, y = point

    # Figure out what ViT subpatch the point belongs to
    p_h, p_w = patch_idx_arr.shape
    p_x, p_y = p_w * x, p_h * y
    p_x_ix, p_y_ix = min(math.floor(p_x), p_w-1), min(math.floor(p_y), p_h-1)
    vit_patch_id = patch_idx_arr[p_y_ix, p_x_ix]

    # Get the corresponding patch id using `pooling_idx`
    token_patch_id = np.argwhere(np.any(pooling_idx == vit_patch_id, -1))
    assert len(token_patch_id) == 1, "Malformed pooling array"
    token_patch_id = token_patch_id[0, 0]

    # Get the subpatch id as the index of `vit_patch_id` within the patches
    # that were pooled for that token
    sub_patch_id = np.argwhere(pooling_idx[token_patch_id] == vit_patch_id)[0, 0]

    # Get the location token if needed
    if location == "3x3":
        in_patch_x = (p_x - p_x_ix)
        in_patch_y = (p_y - p_y_ix)
        location_id = min(math.floor(in_patch_y * 3), 2) + min(math.floor(in_patch_x*3), 2) * 3
        return token_patch_id, sub_patch_id, location_id
    else:
        assert location is None
        return token_patch_id, sub_patch_id


def build_pointing_targets(points, patch_idx_arr, pooling_idx, location=None):
    """Build ground truth target ids for an image"""
    point_target_ids = []
    for point in points:
        point_target_ids.append(get_patch_coordinates(
            point, patch_idx_arr, pooling_idx, location=location))
    return point_target_ids


def build_pointing_targets_video(points, patch_idx_arr, pooling_idx, timestamps, location=None):
    """Build ground truth target ids for a video"""
    point_target_ids = []
    for t, x, y in points:
        frame_ix = np.argmin(np.abs(timestamps - t))
        point_target_ids.append(get_patch_coordinates(
            (x, y), patch_idx_arr[frame_ix], pooling_idx, location=location
        ))
    return point_target_ids


def build_pointing_targets_multi_image(points, tokenized_data: List[TokenizedVisionData], location=None):
    """Build ground truth target ids for set of images"""
    point_target_ids = []
    for t_f, x, y in points:
        t = int(t_f)
        assert t == t_f
        data = tokenized_data[t]
        patch_ids = get_patch_coordinates(
            (x, y), data.token_mapping, data.token_pooling, location=location
        )
        # Offset so its correct for token_ids relative the entire set of images
        token_id = patch_ids[0] + sum(x.token_pooling.shape[0] for x in tokenized_data[:t])
        point_target_ids.append((token_id,) + patch_ids[1:])
    return point_target_ids


EXTRACT_POINT_IDS = re.compile(f"(?:coords=\"|{re.escape(LOCATION_CLS_TOKEN)})([0-9]+)")
NO_POINTS_LABEL = 1000000


@dataclasses.dataclass
class MolmoPointExamplePreprocessor:
    """Preprocessor that combines text with various types of visual input"""

    data_formatter: MolmoPointDataFormatter
    text_preprocessor: MolmoPointInterleavedTextPreprocessor
    for_inference: bool = False
    is_training: bool = False
    image_preprocessor: Any = None
    video_preprocessor: Any = None
    multi_image_preprocessor: Any = None

    include_image: bool = False
    """Output data that can be used for debugging/visualization"""

    patch_location: Optional[str] = None
    """How to build LOCATION tokens, can be None or 3x3"""

    image_pos_ids: str = "one_d"
    """How to build the positions ids of the rotary embeddings"""

    remove_repeats: Optional[str] = None
    """Remove points that end up in the same patch or ViT subpatch"""

    end_of_group_target_id: bool = False
    """Add a PATCH_TOKEN at the end of each group of points with the no-more-points class"""

    sort_points: bool = True
    """Sort points in the order they appear in the input sequence"""

    _output_shapes: Optional[Dict[str, TensorSpec]] = None

    @staticmethod
    def remap_integers(arr):
        mapping = {}
        next_value = 0
        result = []

        for num in arr:
            if num not in mapping:
                mapping[num] = next_value
                next_value += 1
            result.append(mapping[num])
        return np.array(result)

    @staticmethod
    def token_ids_to_coordinates(text, targets, metadata, patch_location: str):
        """Extract points from the model's output

        text: str output of the model, used to extract example ids
        targets: [N, 3] or [N, 2] token index outputs
        metadata: Metadata from the model as produced by `_process_mm_data`
        patch_location: patch location setting
        """
        if targets is None or len(targets) == 0:
            return []
        assert patch_location is None or patch_location == "3x3"
        pooling = metadata["token_pooling"]
        targets = np.asarray(targets)
        targets = targets[targets[:, 0] != NO_POINTS_LABEL]
        vit_patch_ids = pooling[targets[:, 0], targets[:, 1]]
        if vit_patch_ids.ndim == 0:  # numpy might collapse to a scalar if len(targets) == 1
            vit_patch_ids = [vit_patch_ids]
        extracted_points = []
        vit_patch_mapping = metadata["vit_patch_mapping"]
        if isinstance(vit_patch_mapping, list):
            image_sizes = metadata["image_group_size"]
            for patch_ix, patch in enumerate(vit_patch_ids):
                for image_ix, mapping in enumerate(vit_patch_mapping):
                    patch_coords = np.argwhere(mapping == patch)
                    if len(patch_coords) == 1:
                        image_size = image_sizes[image_ix]
                        p_y, p_x = patch_coords[0]
                        if targets.shape[-1] == 3:
                            loc = targets[patch_ix, 2]
                            loc_x = loc // 3
                            loc_y = loc % 3
                            p_x += (loc_x+0.5)*0.33
                            p_y += (loc_y+0.5)*0.33
                        else:
                            p_x += 0.5
                            p_y += 0.5
                        extracted_points.append([
                            image_ix,
                            (p_x / mapping.shape[1]) * image_size[0],
                            (p_y / mapping.shape[0]) * image_size[1],
                        ])
                        break
                else:
                    log.error("Invalid patch id encountered")

        elif "timestamps" in metadata:
            image_size = metadata["image_size"]
            timestamps = metadata["timestamps"]
            extracted_points = []
            for patch_ix, patch in enumerate(vit_patch_ids):
                patch_coords = np.argwhere(vit_patch_mapping == patch)
                if len(patch_coords) == 1:
                    t, p_y, p_x = patch_coords[0]
                    if targets.shape[-1] == 3:
                        loc = targets[patch_ix, 2]
                        loc_x = loc // 3
                        loc_y = loc % 3
                        p_x += (loc_x+0.5)*0.33
                        p_y += (loc_y+0.5)*0.33
                    else:
                        p_x += 0.5
                        p_y += 0.5
                    extracted_points.append([
                        timestamps[t],
                        (p_x / vit_patch_mapping.shape[2]) * image_size[0],
                        (p_y / vit_patch_mapping.shape[1]) * image_size[1],
                    ])
                else:
                    log.error("Invalid patch id encountered")
        else:
            image_size = metadata["image_size"]
            for patch_ix, patch in enumerate(vit_patch_ids):
                patch_coords = np.argwhere(vit_patch_mapping == patch)
                if len(patch_coords) == 1:
                    p_y, p_x = patch_coords[0]
                    if targets.shape[-1] == 3:
                        loc = targets[patch_ix, 2]
                        loc_x = loc // 3
                        loc_y = loc % 3
                        p_x = p_x + (loc_x+0.5)*0.33
                        p_y = p_y + (loc_y+0.5)*0.33
                    else:
                        p_x += 0.5
                        p_y += 0.5
                    extracted_points.append([
                        (p_x / vit_patch_mapping.shape[1]) * image_size[0],
                        (p_y / vit_patch_mapping.shape[0]) * image_size[1],
                    ])
                else:
                    log.error("Invalid patch id encountered")
        if extracted_points:
            if text is not None:
                ids = [int(x) for x in EXTRACT_POINT_IDS.findall(text)]
                if abs(len(ids) - len(extracted_points)) > 1:
                    # We allow a slack of one in case the point just got truncated by the max token limit
                    log.warning(f"Extracted {len(ids)} ids from {text}, but only have {len(extracted_points)} points")
                extracted_points = [[i] + point for i, point in zip(ids, extracted_points)]
            return np.array(extracted_points)
        return []

    @property
    def tokenizer(self):
        return self.text_preprocessor.tokenizer

    @classmethod
    def build(cls, *args, text_seq_len=None, **kwargs) -> 'MolmoPointExamplePreprocessor':
        """
        Build a `MolmoPointExamplePreprocessor` with max_sequence_length defaulting to
        `text_len` + max number of multi-modal vision tokens if text_len is set
        """
        preprocessor = cls(*args, **kwargs)
        if text_seq_len is not None and preprocessor.text_preprocessor.max_sequence_length is None:
            mm_text_len = max(
                x.get_output_shapes()["tokens"].shape[0] for x in
                [preprocessor.image_preprocessor, preprocessor.video_preprocessor, preprocessor.multi_image_preprocessor]
                if x is not None
            )
            max_seq_len = mm_text_len + text_seq_len
            preprocessor.text_preprocessor.max_sequence_length = max_seq_len
        return preprocessor

    def _process_mm_data(self, example, rng) -> Tuple[TokenizedVisionData, Dict]:
        image: Optional[np.ndarray] = None
        image_group: Optional[List[np.ndarray]] = None
        video: Optional[VideoFrames] = None
        if "image" in example:
            is_image_group = isinstance(example["image"], (list, tuple))
            try:
                if is_image_group:
                    image_group = [load_image(x) for x in example["image"]]
                else:
                    image = load_image(example["image"])
            except Exception as e:
                e.add_note(f"Could not load image: {example['image']}")
                raise e
            if not is_image_group:
                # So the formatter can know the height/weight of the video
                example["image"] = image

        if "images" in example:
            example["images"] = [load_image(x) for x in example["images"]]

        if "video" in example:
            if isinstance(example["video"], VideoFrames):
                video = example["video"]
                video_path = None
            else:
                video_path = example["video"]
                try:
                    decode_method = None
                    if "metadata" in example and "decode_method" in example["metadata"]:
                        decode_method = example["metadata"]["decode_method"]
                    clip = None
                    if "metadata" in example and "clip_start_time" in example["metadata"]:
                        clip = (example["metadata"]["clip_start_time"], example["metadata"]["clip_end_time"])
                    subtitle = None
                    if 'subtitle' in example:
                        subtitle = example['subtitle']
                    sampler_overrides = {}
                    if "metadata" in example and "sampler_overrides" in example["metadata"]:
                        sampler_overrides = example["metadata"]["sampler_overrides"]
                    fake_timestamp_fps = None
                    if "metadata" in example and "fake_timestamp_fps" in example["metadata"]:
                        fake_timestamp_fps = example["metadata"]["fake_timestamp_fps"]
                    video = self.video_preprocessor.load_video(
                        example["video"], clip, subtitle=subtitle, decode_method=decode_method, is_training=self.is_training,
                        fake_timestamp_fps=fake_timestamp_fps, **sampler_overrides)
                except Exception as e:
                    e.add_note(f"Could not load video: {example}")
                    raise e
                # So the formatter can know the details of the video
                example["video"] = video

        metadata = {}
        if self.include_image:
            # Include visualization data
            if video is not None:
                if video_path is not None:
                    metadata["video_path"] = video_path
                metadata["video_frames"] = video
            elif image_group is not None:
                metadata["images"] = example["image"]
            elif image is not None:
                metadata["image"] = example["image"]

        tokenized_data: Union[TokenizedVisionData, None, List[TokenizedVisionData]]
        if image is not None:
            if self.image_preprocessor is None:
                raise ValueError("This preprocessor does not support images")
            tokenized_data = self.image_preprocessor(image, is_training=self.is_training, rng=rng)
            h, w = image.shape[:2]
            metadata["image_size"] = (w, h)
            metadata["vit_patch_mapping"] = tokenized_data.token_mapping
        elif video is not None:
            if self.video_preprocessor is None:
                raise ValueError("This preprocessor does not support video")
            tokenized_data = self.video_preprocessor(
                video, is_training=self.is_training, rng=rng)
            h, w = video.frames[0].shape[:2]
            metadata["image_size"] = (w, h)
            metadata["timestamps"] = video.timestamps
            metadata["vit_patch_mapping"] = tokenized_data.token_mapping
        elif image_group is not None:
            if self.multi_image_preprocessor is None:
                raise ValueError("This preprocessor does not support multi-image")
            tokenized_data = self.multi_image_preprocessor(
                image_group, is_training=self.is_training, rng=rng)
            offset = 0
            for tok in tokenized_data:
                tok.token_pooling = np.where(
                    tok.token_pooling >= 0,
                    tok.token_pooling + offset,
                    tok.token_pooling
                )
                tok.token_mapping += offset
                offset += np.prod(tok.images.shape[:2])
            image_sizes = [(x.shape[1], x.shape[0]) for x in image_group]
            metadata["image_group_size"] = image_sizes
            metadata["vit_patch_mapping"] = [x.token_mapping for x in tokenized_data]
        else:
            tokenized_data = None
        return tokenized_data, metadata

    def _build_image_pos_ids(self, example, mm_data) -> np.ndarray:
        if self.image_pos_ids == "one_d":
            if mm_data is None:
                return np.zeros([0, 1], dtype=np.int64)
            # Target IDs are based on indexable tokens only, so the pos ids
            # need to exclude non-indexable image patches
            tokens = example["input_tokens"]
            tokens = tokens[(
                (tokens == self.text_preprocessor.tokenizer.image_patch_token_id)  |
                (tokens == self.text_preprocessor.tokenizer.image_low_res_token_id)
            )]
            assert len(tokens) == example["token_pooling"].shape[-2]
            pos_ids = np.cumsum(tokens == self.text_preprocessor.tokenizer.image_patch_token_id)
            pos_ids = np.maximum(pos_ids-1, 0)  # zero-based indexing
            return pos_ids[:, None]
        elif len(mm_data.token_mapping.shape) == 3:
            ts, h, w = mm_data.token_mapping.shape
            pooling_size = int(np.sqrt(example["token_pooling"].shape[-1]))
            h = (h + pooling_size - 1) // pooling_size
            w = (w + pooling_size - 1) // pooling_size
            assert ts*h*w == example["token_pooling"].shape[0]
            if self.image_pos_ids == "twh":
                coords = np.stack([
                    np.tile(np.arange(ts)[:, None, None], [1, h, w]),
                    np.tile(np.arange(h)[None, :, None], [ts, 1, w]),
                    np.tile(np.arange(w)[None, None, :], [ts, h, 1])
                ], -1)
            elif self.image_pos_ids == "t-wh-ordered":
                ix = np.arange(ts*h*w, dtype=np.int64).reshape(ts, h, w)
                coords = np.stack([
                    np.tile(np.arange(ts)[:, None, None], [1, h, w]),
                    ix, np.transpose(ix, [0, 2, 1])], -1)
            else:
                raise NotImplementedError(self.image_pos_ids)
            return coords.reshape(-1, 3)
        else:
            raise NotImplementedError()

    def __call__(self, example, rng=np.random):
        example = dict(example)
        mm_data, mm_metadata = self._process_mm_data(example, rng)

        # The data formatter needs to know how to sort the points so it gets object ids
        # correct for tracking, and sorting requires doing the point->token id conversion,
        # so we pass through the method to do that
        def _convert_points_to_indices(_points, example_ids=None):
            if mm_data is None:
                raise ValueError("No multi-modal inputs to point to")
            if _points is None or len(_points) == 0:
                out = np.zeros([0, 2 + int(bool(self.patch_location))], dtype=np.int64)
                if example_ids is None:
                    return out
                else:
                    assert len(example_ids) == 0
                    return out, example_ids

            if isinstance(mm_data, list):
                point_target_ids = build_pointing_targets_multi_image(
                    _points, mm_data, self.patch_location)
            elif len(mm_data.token_mapping.shape) == 3:
                point_target_ids = build_pointing_targets_video(
                    _points, mm_data.token_mapping, mm_data.token_pooling,
                    mm_metadata["timestamps"], self.patch_location)
            else:
                point_target_ids = build_pointing_targets(
                    _points, mm_data.token_mapping, mm_data.token_pooling, self.patch_location)

            if example_ids is None:
                if self.remove_repeats == "all":
                    point_target_ids = list(set(point_target_ids))
                elif self.remove_repeats == "subpatch":
                    assert self.patch_location
                    grouped = defaultdict(list)
                    for x in point_target_ids:
                        grouped[x[:2]].append(x[2])
                    point_target_ids = []
                    for (p1, p2), candidates in grouped.items():
                        p3 = np.random.choice(candidates)
                        point_target_ids.append((p1, p2, p3))
                elif self.remove_repeats is not None:
                    raise NotImplementedError(self.remove_repeats)

                if self.sort_points:
                    point_target_ids.sort()
                else:
                    np.random.shuffle(point_target_ids)
                if self.end_of_group_target_id:
                    assert self.patch_location
                    point_target_ids.append([NO_POINTS_LABEL, -1, -1])
                return np.array(point_target_ids)
            else:
                assert len(example_ids) == len(_points)
                if self.sort_points:
                    ix = sorted(range(len(point_target_ids)), key=lambda i: point_target_ids[i])
                    ix = np.array(ix)
                else:
                    ix = np.random.permutation(len(point_target_ids))
                point_target_ids = np.array(point_target_ids)[ix]
                example_ids = np.array(example_ids)[ix]
                # remap ids so object ids are numbered by order of appearance
                # even after the sort, +1 for one based indexing for output object ids
                example_ids = self.remap_integers(example_ids) + 1
                if self.end_of_group_target_id:
                    point_target_ids = np.pad(point_target_ids, [[0, 1], [0, 0]])
                    point_target_ids[-1, 0] = NO_POINTS_LABEL
                    point_target_ids[-1, 1:] = -1
                return point_target_ids, example_ids
        try:
            messages, formatter_metadata = self.data_formatter(
                example, self.is_training, self.for_inference, rng,
                points_to_indices=_convert_points_to_indices
            )
        except Exception as e:
            e.add_note(f"Error formatting example: {example}")
            raise e
        if formatter_metadata is None:
            formatter_metadata = mm_metadata
        else:
            formatter_metadata.update(mm_metadata)

        if isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)
            conversations = messages
        else:
            conversations = [messages]

        weight = example.get("weight")
        if mm_data is None:
            out = self.text_preprocessor.tokenize_and_interleave(conversations, [], weight=weight)

        elif isinstance(mm_data, list):
            multi_model_pos_ids = (
                None if mm_data[0].position_ids is None
                else [mm_data[i].position_ids for i in range(len(mm_data))]
            )
            out = self.text_preprocessor.tokenize_and_interleave(
                conversations,
                [mm_data[i].tokens for i in range(len(mm_data))],
                multi_model_pos_ids,
                weight=weight
            )
            if mm_data[0].images is not None:
                out["images"] = np.concatenate([x.images for x in mm_data], axis=0)
            if mm_data[0].token_pooling is not None:
                out["token_pooling"] = np.concatenate([x.token_pooling for x in mm_data], axis=0)
        else:
            out = self.text_preprocessor.tokenize_and_interleave(
                conversations,
                [mm_data.tokens],
                None if mm_data.position_ids is None else [mm_data.position_ids],
                weight
            )
            if mm_data.images is not None:
                out["images"] = mm_data.images
            if mm_data.token_pooling is not None:
                out["token_pooling"] = mm_data.token_pooling

        formatter_metadata["token_pooling"] = out.get("token_pooling")
        out["image_pos_ids"] = self._build_image_pos_ids(out, mm_data)

        metadata = example.get("metadata", {})
        metadata.update(formatter_metadata)
        out["metadata"] = metadata
        return out

    def get_output_shapes(self) -> Dict[str, TensorSpec]:
        if self._output_shapes is not None:
            return self._output_shapes
        specs = [
            x.get_output_shapes() for x in
            [self.image_preprocessor, self.video_preprocessor, self.multi_image_preprocessor]
            if x is not None
        ]
        spec = TensorSpec.max_dictionaries(*specs)
        max_seq_len = self.text_preprocessor.max_sequence_length
        if max_seq_len:
            if spec["tokens"].shape[0] > max_seq_len:
                raise ValueError(f"Max sequence length {spec['tokens'].shape[0]} is greater than preprocessor max token length {max_seq_len}")
            spec["tokens"] = TensorSpec([max_seq_len], np.int64)
        else:
            # Unknown since we don't have a bound on the number of tokens
            spec["tokens"] = TensorSpec([None], np.int64)
        n_point_targets = 2
        if self.patch_location:
            n_point_targets += 1
        spec["point_target_ids"] = VariablePaddingSpec([1, n_point_targets], np.int64)
        if self.image_pos_ids:
            if self.image_pos_ids == "one_d":
                spec["image_pos_ids"] = VariablePaddingSpec([1, 1], np.int64)
            else:
                spec["image_pos_ids"] = VariablePaddingSpec([1, 3], np.int64)
        self._output_shapes = spec
        return spec

