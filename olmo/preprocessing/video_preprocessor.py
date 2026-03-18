from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from olmo.config import D
from olmo.preprocessing.image_preprocessor import ImagePreprocessor
from olmo.preprocessing.preprocessor_utils import TensorSpec, TokenizedVisionData, \
    batch_pixels_to_patches, arange_for_pooling
from olmo.tokenizer import HfTokenizerWrapper

from olmo.data.video_loader import VideoLoaderConfig, VideoLoader, VideoFrames


@dataclass
class VideoPreprocessorConfig(VideoLoaderConfig):
    """Video preprocessor, converts videos into crops and tokens"""

    time_mode: str = "per-frame-compact"

    subtitle_mode: str = "frame_1"  # "frame_N", "all", "truncate_N", "ignore"

    pooling_w: int = 3
    """pooling w stride"""

    pooling_h: int = 3
    """pooling h stride"""

    use_frame_special_tokens: bool = True
    """Whether to use frame special tokens in the video preprocessor"""

    per_frame_special_token: bool = False
    """Put special tokens around each frame instead of around all the frames"""

    max_subtitle_tokens: Optional[int] = None
    """Max number of subtitle tokens to use"""

    @classmethod
    def build_from_legacy_config(cls, config: D) -> D:
        assert config.max_crops == 1
        assert config.periodic_high_res_frame is None
        assert not config.image_padding_mask
        assert not config.query_based_resolution_selection
        return VideoPreprocessorConfig(
            max_frames=config.max_frames,
            frame_sample_mode=config.frame_sample_mode,
            candidate_sampling_fps=config.candidate_sampling_fps,
            cache_videos=config.cache_videos,
            loading_method=config.loading_method,
            max_fps=config.max_fps,
            time_sampling=config.time_sampling,
            time_mode=config.time_mode,
            subtitle_mode=config.subtitle_mode,
            pooling_w=config.pooling_w,
            pooling_h=config.pooling_h,
            use_frame_special_tokens=config.use_frame_special_tokens,
            per_frame_special_token=True,
            max_subtitle_tokens=config.max_subtitle_tokens,
        )

    def build_video_preprocessor(self, tokenizer, image_preprocessor, add_end_of_mm_token=False) -> 'TokenIndexingVideoPreprocessor':
        assert self.pooling_w == self.pooling_h
        return TokenIndexingVideoPreprocessor(
            tokenizer,
            image_preprocessor,
            video_loader=self.build_video_loader(),
            pooling=self.pooling_w,
            time_mode=self.time_mode,
            subtitle_mode=self.subtitle_mode,
            use_frame_special_tokens=self.use_frame_special_tokens,
            max_subtitle_tokens=self.max_subtitle_tokens,
            per_frame_special_token=self.per_frame_special_token,
        )


@dataclass
class TokenIndexingVideoPreprocessor:
    tokenizer: HfTokenizerWrapper
    image_preprocessor: ImagePreprocessor
    video_loader: VideoLoader
    pooling: int = 3
    time_mode: str = "per-frame-compact"
    use_frame_special_tokens: bool = False
    subtitle_mode: str = "frame_3"  # "frame_N", "all", "truncate_N", "ignore"
    max_subtitle_tokens: Optional[int] = None
    subtitle_time_window: int = None
    per_frame_special_token: bool = False

    def __post_init__(self):
        if self.subtitle_mode.startswith("truncate"):
            assert self.max_subtitle_tokens is None
            self.max_subtitle_tokens = int(self.subtitle_mode.split("_")[1])
            self.subtitle_mode = "truncate"

        if self.subtitle_mode.startswith("frame_"):
            self.subtitle_time_window = int(self.subtitle_mode.split("_")[1])
            self.subtitle_mode = "frame"
        else:
            self.subtitle_time_window = 0

    @property
    def max_frames(self) -> int:
        return self.video_loader.sampler.max_frames

    def load_video(self, *args, **kwargs):
        return self.video_loader(*args, **kwargs)

    def get_output_shapes(self):
        h, w = self.image_preprocessor.base_image_input_size
        fake_input = VideoFrames(
            np.zeros([self.max_frames, h, w, 3], dtype=np.uint8),
            timestamps=np.arange(self.max_frames)*32.5,
            target_fps=None
        )
        out = self(fake_input, ["fake query"])
        return TensorSpec.get_spec(out)

    def video_to_patches_and_tokens(
        self,
        frames,
        frame_prefixes: List[List[int]],
        is_training=False,
        rng=None,
    ):
        base_image_input_size = self.image_preprocessor.base_image_input_size
        image_patch_size = self.image_preprocessor.image_patch_size
        images = frames.frames
        n_frames = len(images)

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = images.shape[:2]
        crop_size = base_image_input_size[0]

        resized, _, patch_idx = self.image_preprocessor.build_single_crop(
            images, is_training=is_training, rng=rng)

        patches = np.prod(patch_idx.shape)
        idx = arange_for_pooling(patch_idx, self.pooling, self.pooling)
        idx = np.tile(np.expand_dims(idx, axis=0), [n_frames, 1, 1, 1])
        high_res_idx = np.where(
            idx < 0,
            idx,
            idx + (patches * np.arange(n_frames))[:, None, None, None]
        )
        high_h, high_w = high_res_idx.shape[1:3]

        # Each frame contributes high_h*high_w pooling rows and 1 image
        pooling_per_frame = high_h * high_w
        cum_token_pooling_bounds = np.cumsum(np.full(n_frames, pooling_per_frame)).astype(np.int64)
        cum_image_bounds = np.arange(1, n_frames + 1, dtype=np.int64)

        if self.use_frame_special_tokens:
            start_tok, end_tok = self.tokenizer.frame_start_token_id, self.tokenizer.frame_end_token_id
        else:
            start_tok, end_tok = self.tokenizer.image_start_token_id, self.tokenizer.image_end_token_id

        tokens = []
        if not self.per_frame_special_token:
            tokens.append([start_tok])
        for frame in range(n_frames):
            tokens.append(frame_prefixes[frame])
            if self.per_frame_special_token:
                tokens.append([start_tok])
            tokens.append(np.full([high_h*high_w], self.tokenizer.image_patch_token_id))
            if self.per_frame_special_token:
                tokens.append([end_tok])
        if not self.per_frame_special_token:
            tokens.append([end_tok])
        return TokenizedVisionData(
            tokens=np.concatenate(tokens),
            images=batch_pixels_to_patches(resized, image_patch_size),
            token_pooling=high_res_idx.reshape(high_h*high_w*n_frames, -1),
            token_mapping=np.arange(n_frames*crop_patch_h*crop_patch_w).reshape(n_frames, crop_patch_h, crop_patch_w),
        )

    def __call__(
        self,
        video_frames: VideoFrames,
        is_training=False,
        rng=None,
        metadata=None
    ) -> TokenizedVisionData:
        tok = self.tokenizer

        frame_prefixes = []
        for frame_idx, frame_time in enumerate(video_frames.timestamps):
            if self.time_mode == "none":
                frame_prefixes.append([])
            else:
                if self.time_mode == "numbered-frames":
                    prefix = f"{frame_idx+1}:"
                elif self.time_mode == "per-frame":
                    prefix = f"time {frame_time:.2f}"
                elif self.time_mode == "per-frame-compact":
                    prefix = f"{frame_time:.1f}"
                else:
                    raise NotImplementedError()
                prev_space = " " if frame_idx > 0 else ""
                frame_prefixes.append(self.tokenizer.encode(prev_space + prefix + " "))

        data = self.video_to_patches_and_tokens(video_frames, frame_prefixes, is_training, rng)

        if video_frames.subtitle:
            subtitle = video_frames.subtitle
            if subtitle is not None and self.subtitle_mode != 'ignore':
                if isinstance(subtitle, str):
                    subtitle_str = "subtitle\n" + subtitle
                elif isinstance(subtitle, dict):
                    subtitle_str = "subtitle\n"
                    for (s, e), txt in subtitle.items():
                        if self.subtitle_mode == "frame":
                            for t in video_frames.timestamps:
                                if not (e < (t - self.subtitle_time_window) or (t + self.subtitle_time_window) < s):
                                    subtitle_str += f"{s:.1f} - {e:.1f} {txt}\n"
                                    break
                        else:
                            subtitle_str += f"{s:.1f} - {e:.1f} {txt}\n"
                else:
                    raise ValueError("Subtitle must be a string or a dict")

                subtitle_tokens = self.tokenizer.encode(subtitle_str)
                if self.max_subtitle_tokens is not None:
                    if len(subtitle_tokens) > self.max_subtitle_tokens:
                        subtitle_tokens = subtitle_tokens[:self.max_subtitle_tokens]
                data.tokens = np.concatenate([data.tokens, subtitle_tokens], -1)
        return data
