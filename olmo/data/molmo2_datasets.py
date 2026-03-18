import atexit
import json
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from os.path import join, exists
from typing import Optional, Tuple

import numpy as np
import datasets

from olmo.data.dataset import DatasetBase, DATA_HOME, VIDEO_DATA_HOME, Dataset
from olmo.util import set_example_style
from olmo.data.dataset import DatasetBase, DATA_HOME, VIDEO_DATA_HOME
from olmo.util import set_example_style, split_into_groups


if DATA_HOME:
    VIDEO_HOME = join(DATA_HOME, "videos")
else:
    VIDEO_HOME = None


def _decode_rle_mask(rle):
    """Decode an RLE-encoded mask into a 2D numpy array.

    Handles both compressed (string counts via pycocotools) and
    uncompressed (list-of-int counts) COCO RLE formats.
    """
    h, w = rle['size']
    counts = rle['counts']

    if isinstance(counts, str):
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle)

    # Uncompressed RLE: alternating runs starting with 0s
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # odd indices are 1-runs
            mask[pos:pos + count] = 1
        pos += count
    return mask.reshape((h, w), order='F')  # COCO uses column-major order


def _load_hf_dataset(hf_source, split, local_name, config=None):
    """Load an HF dataset, caching locally under VIDEO_DATA_HOME.

    If a local copy exists at VIDEO_DATA_HOME/{local_name}, loads from disk.
    Otherwise downloads from HF and saves locally.
    """
    local_dir = join(VIDEO_DATA_HOME, local_name) if VIDEO_DATA_HOME else None

    if local_dir and exists(local_dir):
        logging.info(f"Loading {hf_source} config={config} split={split} from {local_dir}")
        return datasets.load_from_disk(local_dir)

    logging.info(f"Downloading {hf_source} config={config} split={split} from HuggingFace")
    ds = datasets.load_dataset(hf_source, config, split=split)

    if local_dir:
        logging.info(f"Saving to {local_dir}")
        ds.save_to_disk(local_dir)

    return ds

def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60  # Keep decimal
    # keep two decimal places for seconds by default
    formatted = f"{hours:02}:{minutes:02}:{seconds:05.2f}"
    return formatted


def sample_random_clip(
    video_duration: float,
    start_time: float,
    end_time: float,
    min_seconds: float,
    max_seconds: float,
    timestamp_step: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Randomly choose a clip [clip_start, clip_end] such that:
      - 0 <= clip_start <= start_time <= end_time <= clip_end <= video_duration
      - min_seconds <= (clip_end - clip_start) <= max_seconds
      - clip_start is a multiple of timestamp_step (e.g., 0.5, 1/30, etc.)
      - Uniform randomness by default:
          * start is uniform over all feasible aligned starts
          * duration is uniform within the feasible range for that start

    Args:
      timestamp_step: grid step for clip_start alignment (must be > 0).
      seed: optional RNG seed for reproducibility.

    Raises:
      ValueError if inputs are invalid or constraints are impossible.
    """
    EPS = 1e-9

    # Validation
    if video_duration <= 0:
        raise ValueError("video_duration must be positive.")
    if min_seconds <= 0 or max_seconds <= 0:
        raise ValueError("min_seconds and max_seconds must be positive.")
    if timestamp_step <= 0:
        raise ValueError("timestamp_step must be positive.")
    if min_seconds - max_seconds > EPS:
        raise ValueError("min_seconds cannot exceed max_seconds.")
    if not (0 <= start_time <= end_time <= video_duration):
        raise ValueError(f"Require 0 <= start_time <= end_time <= video_duration but got {start_time, end_time, video_duration}.")

    seg_len = end_time - start_time
    if seg_len - max_seconds > EPS:
        raise ValueError(f"Required segment is longer than max_seconds. Got {start_time, end_time, max_seconds}")

    # Global feasible duration range
    W_min = max(seg_len, min_seconds)
    W_max = min(max_seconds, video_duration)
    if W_min - W_max > EPS:
        raise ValueError("No feasible clip length given the constraints and video length.")

    step = float(timestamp_step)

    # k range from lower bound (ensuring end_time can be included with max W)
    start_lower = max(0.0, end_time - W_max)
    # and upper bound (must not exceed start_time, and leave room for at least W_min)
    start_upper = min(start_time, video_duration - W_min)

    k_min = math.ceil((start_lower - EPS) / step)
    k_max = math.floor((start_upper + EPS) / step)

    if k_min > k_max:
        raise ValueError("No grid-aligned start can satisfy the constraints.")

    # Collect feasible k where the local W interval is non-empty
    feasible_k = []
    for k in range(k_min, k_max + 1):
        clip_start = k * step
        w_low = max(W_min, end_time - clip_start)
        w_high = min(W_max, video_duration - clip_start)
        if w_low <= w_high + EPS:
            feasible_k.append((k, w_low, w_high))

    if not feasible_k:
        raise ValueError("No grid-aligned start yields a feasible duration window.")

    rng = random.Random(seed) if seed is not None else random

    # Sample start index uniformly among feasible grid points
    k, w_low, w_high = rng.choice(feasible_k)
    clip_start = k * step

    # Sample duration uniformly within feasible window for this start
    W = w_low if abs(w_high - w_low) <= EPS else rng.uniform(w_low, w_high)
    clip_end = clip_start + W

    # Final safety checks (tolerant to float noise)
    assert -EPS <= clip_start <= start_time + EPS
    assert end_time - EPS <= clip_end <= video_duration + EPS
    assert min_seconds - EPS <= (clip_end - clip_start) <= max_seconds + EPS
    assert abs((clip_start / step) - round(clip_start / step)) <= 1e-6  # grid aligned

    return float(clip_start), float(clip_end)


class Molmo2CaptionsEval(DatasetBase):
    HF_SOURCE = "allenai/Molmo2-CapEval"
    HF_SPLIT = "test"
    LOCAL_NAME = "Molmo2-CapEval"
    VIDEO_DIR = join(VIDEO_DATA_HOME, "video-caption-eval") if VIDEO_DATA_HOME else None
    VIMEO_HF_REPO = "allenai/Molmo2-CapEval"
    VIMEO_ZIP_NAME = "vimeo_videos.zip"
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "bdd100k and ego4d videos must be downloaded manually.\n"
        "Place them under MOLMO_DATA_DIR/video_datasets/video-caption-eval/ so the structure is:\n"
        "  video-caption-eval/bdd100k/{video_id}.mov\n"
        "  video-caption-eval/ego4d/{video_id}.mp4\n"
        "\n"
        "For ego4d, some videos need to be clipped to a specific time range.\n"
        "If a row in the dataset has video_start and video_end set, clip the original\n"
        "ego4d video using ffmpeg and save with this naming convention:\n"
        "  ffmpeg -ss {start} -to {end} -i {video_id}.mp4 -c copy {video_id}_bounded_decimal_2_{start}_{end}.mp4\n"
        "For example, a video 0ae6293e-...-e8f27fcde953 clipped from 0.17s to 141.74s becomes:\n"
        "  ego4d/0ae6293e-...-e8f27fcde953_bounded_decimal_2_0.17_141.74.mp4\n"
        "Note: trailing zeros in timestamps are stripped (e.g. 0.80 -> 0.8, 3.00 -> 3.0)."
    )

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    @classmethod
    def _ensure_videos(cls):
        """Auto-download vimeo videos from HF; warn about manual downloads for others."""
        if cls.VIDEO_DIR is None:
            raise RuntimeError("MOLMO_DATA_DIR is not set. Cannot determine VIDEO_DIR.")
        os.makedirs(cls.VIDEO_DIR, exist_ok=True)
        vimeo_dir = join(cls.VIDEO_DIR, "vimeo")
        if not exists(vimeo_dir):
            import zipfile
            from huggingface_hub import hf_hub_download
            logging.info(f"Downloading vimeo videos from {cls.VIMEO_HF_REPO}...")
            zip_path = hf_hub_download(
                repo_id=cls.VIMEO_HF_REPO,
                filename=cls.VIMEO_ZIP_NAME,
                repo_type="dataset",
                local_dir=cls.VIDEO_DIR,
            )
            logging.info(f"Extracting {zip_path} to {cls.VIDEO_DIR}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(cls.VIDEO_DIR)
            os.remove(zip_path)
            logging.info("Vimeo video extraction complete.")
        for src in ("bdd100k", "ego4d"):
            if not exists(join(cls.VIDEO_DIR, src)):
                logging.warning(f"{src}/ not found in {cls.VIDEO_DIR}. {cls.MANUAL_DOWNLOAD_INSTRUCTIONS}")

    def _build_video_index(self):
        """Walk VIDEO_DIR once and build a mapping from filename stem to full path."""
        index = {}
        for root, dirs, files in os.walk(self.VIDEO_DIR):
            for f in files:
                stem = os.path.splitext(f)[0]
                index[stem] = join(root, f)
        return index

    def _find_video_path(self, video_id, source, video_start=None, video_end=None):
        """Look up video path from the pre-built index."""
        # ego4d bounded clips
        if source == "ego4d" and video_start is not None and video_end is not None:
            start_str = f"{video_start:.2f}".rstrip("0") or "0"
            end_str = f"{video_end:.2f}".rstrip("0") or "0"
            if start_str.endswith("."):
                start_str += "0"
            if end_str.endswith("."):
                end_str += "0"
            bounded_id = f"{video_id}_bounded_decimal_2_{start_str}_{end_str}"
            if bounded_id in self._video_index:
                return self._video_index[bounded_id]

        # Direct lookup by video_id
        if video_id in self._video_index:
            return self._video_index[video_id]

        # vimeo files are prefixed with "vimeo_"
        if source == "vimeo":
            vimeo_id = f"vimeo_{video_id}"
            if vimeo_id in self._video_index:
                return self._video_index[vimeo_id]

        raise FileNotFoundError(
            f"Video not found for video_id={video_id}, source={source}. "
            f"Looked in: {self.VIDEO_DIR}\n"
        )

    def __init__(self, split):
        assert split in ["test"]
        super().__init__(split)

    def load(self):
        self._ensure_videos()
        self._video_index = self._build_video_index()
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)
        data = []
        for row in ds:
            video_path = self._find_video_path(
                row["video_id"], row["source"],
                video_start=row["video_start"], video_end=row["video_end"],
            )
            example = {
                "video": video_path,
                "style": "video_long_caption",
                "metadata": {
                    "video_path": video_path,
                    "video_id": row["video_id"],
                    "source": row["source"],
                    "clip_start_time": row["video_start"],
                    "clip_end_time": row["video_end"],
                    "duration": row["duration"],
                    "data": {
                        "atomic_statements"     : row["atomic_statements"],
                        "categories"            : row["statement_categories"],
                        "aggregated_annotations": row["aggregated_caption"],
                    }
                },
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return self.data[idx]


YOUTUBE_SUBDIRS = [
    "youtube-cc-exist", "youtube-cc-kw", "youtube-cc-temporal",
]

VIDEO_POINT_SUBDIRS = {
    "youtube": [
        "youtube-cc/youtube-cc-exist-2fps",
        "youtube-cc/youtube-cc-kw-2fps",
        "youtube-cc/youtube-cc-temporal-2fps",
    ],
    "MammalNet": ["MammalNet/trimmed_video-2fps"],
    "generated": ["generated_videos"],
}

VIDEO_CAP_SUBDIRS = [
    "youtube-cc/youtube-cc-exist", "youtube-cc/youtube-cc-kw", "youtube-cc/youtube-cc-temporal",
    "youtube_temporal/batch_1", "intern/batch_1", "kw_youtube/batch_1",
    # LLaVA-Video-178K subdirs (academic_source uses v_ prefix, liwei uses ytb_ prefix)
    "LLaVA-Video-178K/0_30_s_academic_v0_1/academic_source/activitynet",
    "LLaVA-Video-178K/30_60_s_academic_v0_1/academic_source/activitynet",
    "LLaVA-Video-178K/1_2_m_academic_v0_1/academic_source/activitynet",
    "LLaVA-Video-178K/2_3_m_academic_v0_1/academic_source/activitynet",
    "LLaVA-Video-178K/0_30_s_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
    "LLaVA-Video-178K/30_60_s_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
    "LLaVA-Video-178K/1_2_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
    "LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
    "LLaVA-Video-178K/1_2_m_youtube_v0_1/liwei_youtube_videos/videos/hdvila",
]

# Subdirs where videos are stored as {subdir}/{video_id}/{video_file}
# (as opposed to flat file-based {subdir}/{video_id}.ext)
_FOLDER_BASED_SUBDIRS = {
    "youtube-cc/youtube-cc-exist", "youtube-cc/youtube-cc-kw", "youtube-cc/youtube-cc-temporal",
}

# Koala metadata: downloaded from HF and extracted under VIDEO_DATA_HOME
_KOALA_META_HF_REPO = "allenai/Molmo2-Cap"
_KOALA_META_ZIP_NAME = "koala_meta.zip"
_KOALA_META_DIR = join(VIDEO_DATA_HOME, "koala_meta") if VIDEO_DATA_HOME else None

_koala_yt_to_filename = None  # Lazy-loaded: {youtube_id: koala_filename}


def _download_koala_meta():
    """Download and extract koala_meta.zip from HF if not already present."""
    if _KOALA_META_DIR is None:
        raise RuntimeError("MOLMO_DATA_DIR is not set. Cannot determine koala metadata dir.")
    if os.path.isdir(_KOALA_META_DIR):
        return
    import zipfile
    from huggingface_hub import hf_hub_download
    logging.info(f"Downloading koala metadata from {_KOALA_META_HF_REPO}...")
    zip_path = hf_hub_download(
        repo_id=_KOALA_META_HF_REPO,
        filename=_KOALA_META_ZIP_NAME,
        repo_type="dataset",
        local_dir=VIDEO_DATA_HOME,
    )
    logging.info(f"Extracting {zip_path} to {VIDEO_DATA_HOME}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(VIDEO_DATA_HOME)
    os.remove(zip_path)
    logging.info("Koala metadata extraction complete.")


def _get_koala_lookup():
    """Lazy-load koala YouTube ID -> filename mapping from metadata JSONs."""
    global _koala_yt_to_filename
    if _koala_yt_to_filename is not None:
        return _koala_yt_to_filename

    _download_koala_meta()

    _koala_yt_to_filename = {}
    if not os.path.isdir(_KOALA_META_DIR):
        logging.warning(f"Koala metadata dir not found: {_KOALA_META_DIR}")
        return _koala_yt_to_filename

    for fname in os.listdir(_KOALA_META_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(join(_KOALA_META_DIR, fname)) as f:
                meta = json.load(f)
            url = meta.get("url", "")
            if "v=" in url:
                yt_id = url.split("v=", maxsplit=1)[1]
                koala_name = fname.replace(".json", "")
                _koala_yt_to_filename.setdefault(yt_id, koala_name)
        except Exception as e:
            logging.warning(f"Failed to read koala metadata {fname}: {e}")

    logging.info(f"Loaded {len(_koala_yt_to_filename)} koala YouTube ID mappings")
    return _koala_yt_to_filename


def _get_cache_path(video_dir, cache_name="youtube_cc.pkl"):
    """Get cache file path for a video directory."""
    if VIDEO_DATA_HOME is None:
        return None
    cache_dir = join(VIDEO_DATA_HOME, ".molmo2_indices")
    os.makedirs(cache_dir, exist_ok=True)
    return join(cache_dir, cache_name)


def _load_cache_from_disk(cache_path):
    """Load video path cache from disk."""
    if not exists(cache_path):
        return {}
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logging.info(f"Loaded {len(cache)} video paths from cache: {cache_path}")
        return cache
    except Exception as e:
        logging.warning(f"Failed to load cache {cache_path}: {e}")
        return {}


def _save_cache_to_disk(cache_path, cache):
    """Save video path cache to disk."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        logging.debug(f"Saved {len(cache)} video paths to cache: {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save cache {cache_path}: {e}")


# Module-level cache: shared across all QA dataset instances in the process
_youtube_cc_index_cache = {}  # {video_dir: {video_id: video_path}}
_cache_modified = set()  # Track which caches have been modified


def _save_all_caches():
    """Save all modified caches to disk on exit."""
    for video_dir in _cache_modified:
        cache_path = _get_cache_path(video_dir)
        if cache_path and video_dir in _youtube_cc_index_cache:
            _save_cache_to_disk(cache_path, _youtube_cc_index_cache[video_dir])


atexit.register(_save_all_caches)


def _find_video_by_id(video_dir, video_id, download_instructions):
    """
    Find video path for a video_id using on-demand lookup.
    Checks in-memory cache first, then disk cache, then searches 3 subdirectories.
    Discovered paths are cached in memory and persisted to disk on exit.
    """
    # Initialize cache for this video_dir if needed (load from disk)
    if video_dir not in _youtube_cc_index_cache:
        cache_path = _get_cache_path(video_dir)
        if cache_path:
            _youtube_cc_index_cache[video_dir] = _load_cache_from_disk(cache_path)
        else:
            _youtube_cc_index_cache[video_dir] = {}

    # Check in-memory cache
    cache = _youtube_cc_index_cache[video_dir]
    if video_id in cache:
        return join(video_dir, cache[video_id])

    # On-demand lookup: check each subdir for this video_id
    for subdir in YOUTUBE_SUBDIRS:
        video_folder = join(video_dir, subdir, video_id)
        if exists(video_folder):
            try:
                for f in os.listdir(video_folder):
                    if f.endswith((".mp4", ".mov", ".mkv", ".webm")):
                        rel_path = join(subdir, video_id, f)
                        # Cache the relative path for portability
                        cache[video_id] = rel_path
                        _cache_modified.add(video_dir)
                        return join(video_dir, rel_path)
            except Exception as e:
                logging.warning(f"Error reading {video_folder}: {e}")

    raise FileNotFoundError(
        f"Video not found for video_id={video_id}. "
        f"Looked in: {video_dir}/{{youtube-cc-exist,youtube-cc-kw,youtube-cc-temporal}}/{video_id}/\n"
        f"{download_instructions}"
    )


_video_point_index_cache = {}  # {video_dir: {video_id: rel_path}}
_video_point_cache_modified = set()


def _save_all_video_point_caches():
    """Save all modified video point caches to disk on exit."""
    for video_dir in _video_point_cache_modified:
        cache_path = _get_cache_path(video_dir, cache_name="video_point.pkl")
        if cache_path and video_dir in _video_point_index_cache:
            _save_cache_to_disk(cache_path, _video_point_index_cache[video_dir])


atexit.register(_save_all_video_point_caches)


def _find_video_point_path_by_id(video_dir, video_id, video_source):
    """Find 2fps video path for a video_id using on-demand lookup.

    Checks in-memory cache first, then disk cache, then searches subdirectories.
    Discovered paths are cached in memory and persisted to disk on exit.

    Looks for video_dir/subdir/{video_id}_2fps.mp4 across subdirs matching video_source.
    Falls back to searching all subdirs if video_source is not in VIDEO_POINT_SUBDIRS.

    For video IDs containing '/' (e.g. 'sora2/some_video'), the prefix is treated as a
    sub-subdirectory with '-2fps' appended (e.g. 'video_dir/sora2-2fps/some_video_2fps.mp4').
    """
    VIDEO_DOWNLOAD_INSTRUCTIONS = (
        "See the HF repo README for download instructions:\n"
        "  https://huggingface.co/datasets/allenai/Molmo2-VideoPoint\n"
        "Place downloaded videos under MOLMO_DATA_DIR/video_datasets/ and convert them to 2fps version so the structure is:\n"
        "  youtube-cc/youtube-cc-exist-2fps\n"
        "  youtube-cc/youtube-cc-kw-2fps\n"
        "  youtube-cc/youtube-cc-temporal-2fps\n"
        "  MammalNet/trimmed_video-2fps\n"
        "  generated_videos\n"
    )

    # Initialize cache for this video_dir if needed (load from disk)
    if video_dir not in _video_point_index_cache:
        cache_path = _get_cache_path(video_dir, cache_name="video_point.pkl")
        if cache_path:
            _video_point_index_cache[video_dir] = _load_cache_from_disk(cache_path)
        else:
            _video_point_index_cache[video_dir] = {}

    # Check in-memory cache
    cache = _video_point_index_cache[video_dir]
    if video_id in cache:
        return join(video_dir, cache[video_id])

    # On-demand lookup: check subdirs for this video_id
    subdirs = VIDEO_POINT_SUBDIRS.get(video_source)
    if subdirs is None:
        subdirs = [s for group in VIDEO_POINT_SUBDIRS.values() for s in group]

    if "/" in video_id:
        # e.g. video_id="sora2/some_video" -> subdir_suffix="sora2-2fps", filename="some_video"
        prefix, filename = video_id.rsplit("/", 1)
        subdir_suffix = f"{prefix}-2fps"
        for subdir in subdirs:
            rel_path = join(subdir, subdir_suffix, f"{filename}_2fps.mp4")
            if exists(join(video_dir, rel_path)):
                cache[video_id] = rel_path
                _video_point_cache_modified.add(video_dir)
                return join(video_dir, rel_path)
    else:
        for subdir in subdirs:
            rel_path = join(subdir, f"{video_id}_2fps.mp4")
            if exists(join(video_dir, rel_path)):
                cache[video_id] = rel_path
                _video_point_cache_modified.add(video_dir)
                return join(video_dir, rel_path)

    searched = ", ".join(subdirs)
    raise FileNotFoundError(
        f"Video not found for video_id={video_id}, video_source={video_source}.\n"
        f"Searched in: {searched}\n"
        "Please follow download instructions below:\n"
        f"{VIDEO_DOWNLOAD_INSTRUCTIONS}"
    )


_video_cap_index_cache = {}  # {video_dir: {video_id: rel_path}}
_video_cap_cache_modified = set()


def _save_all_video_cap_caches():
    """Save all modified video cap caches to disk on exit."""
    for video_dir in _video_cap_cache_modified:
        cache_path = _get_cache_path(video_dir, cache_name="video_cap.pkl")
        if cache_path and video_dir in _video_cap_index_cache:
            _save_cache_to_disk(cache_path, _video_cap_index_cache[video_dir])


atexit.register(_save_all_video_cap_caches)


def _find_video_cap_path_by_id(video_dir, video_id):
    """Find video path for a Molmo2-Cap video_id.

    Searches VIDEO_CAP_SUBDIRS under video_dir. Handles three layouts:
      - Folder-based (youtube-cc-*): video_dir/subdir/{video_id}/{video_file}
      - File-based (intern, kw_youtube, youtube_temporal): video_dir/subdir/{video_id}.{ext}
      - LLaVA (activitynet uses v_ prefix, liwei uses ytb_ prefix): video_dir/subdir/{prefix}{video_id}.{ext}
      - Koala: video_id is a YouTube ID, mapped to koala filename via metadata

    Results are cached in memory and persisted to disk on exit.
    """
    VIDEO_DOWNLOAD_INSTRUCTIONS = (
        "See the HF repo README for download instructions:\n"
        "  https://huggingface.co/datasets/allenai/Molmo2-Cap\n"
        "Videos are searched under MOLMO_DATA_DIR/video_datasets/ in:\n"
        "  youtube-cc/{youtube-cc-exist,youtube-cc-kw,youtube-cc-temporal}/{video_id}/{video_file}\n"
        "  {intern,kw_youtube,youtube_temporal}/batch_1/{video_id}.{ext}\n"
        "  koala/batch_1/{koala_filename}.mp4 (mapped via koala metadata)\n"
        "  LLaVA-Video-178K/.../{v_,ytb_}{video_id}.{ext}"
    )
    if video_dir not in _video_cap_index_cache:
        cache_path = _get_cache_path(video_dir, cache_name="video_cap.pkl")
        if cache_path:
            _video_cap_index_cache[video_dir] = _load_cache_from_disk(cache_path)
        else:
            _video_cap_index_cache[video_dir] = {}

    cache = _video_cap_index_cache[video_dir]
    if video_id in cache:
        return join(video_dir, cache[video_id])

    video_exts = (".mp4", ".mov", ".mkv", ".webm")

    # Build list of (subdir, filename_without_ext) candidates to search
    candidates = []
    for subdir in VIDEO_CAP_SUBDIRS:
        if "activitynet" in subdir:
            # LLaVA activitynet: stored as v_{video_id}.ext
            candidates.append((subdir, f"v_{video_id}"))
        elif "youtube_video_2024" in subdir:
            # LLaVA liwei youtube: stored as ytb_{video_id}.ext
            candidates.append((subdir, f"ytb_{video_id}"))
        elif "hdvila" in subdir:
            # LLaVA hdvila: stored as {video_id}.ext (no prefix)
            candidates.append((subdir, video_id))
        else:
            candidates.append((subdir, video_id))

    # Also try koala: video_id is a YouTube ID, need to map to koala filename
    koala_lookup = _get_koala_lookup()
    koala_filename = koala_lookup.get(video_id)
    if koala_filename:
        candidates.append(("koala/batch_1", koala_filename))

    for subdir, filename in candidates:
        if subdir in _FOLDER_BASED_SUBDIRS:
            # Folder-based: subdir/{video_id}/{video_file}
            video_folder = join(video_dir, subdir, filename)
            if exists(video_folder):
                try:
                    for f in os.listdir(video_folder):
                        if f.endswith(video_exts):
                            rel_path = join(subdir, filename, f)
                            cache[video_id] = rel_path
                            _video_cap_cache_modified.add(video_dir)
                            return join(video_dir, rel_path)
                except Exception as e:
                    logging.warning(f"Error reading {video_folder}: {e}")
        else:
            # File-based: subdir/{filename}.ext
            subdir_path = join(video_dir, subdir)
            if not exists(subdir_path):
                continue
            for ext in video_exts:
                rel_path = join(subdir, f"{filename}{ext}")
                if exists(join(video_dir, rel_path)):
                    cache[video_id] = rel_path
                    _video_cap_cache_modified.add(video_dir)
                    return join(video_dir, rel_path)

    searched = ", ".join(VIDEO_CAP_SUBDIRS + (["koala/batch_1"] if koala_filename else []))
    raise FileNotFoundError(
        f"Video not found for video_id={video_id}.\n"
        f"Searched in: {searched} under {video_dir}\n"
        "Please follow download instructions below:\n"
        f"{VIDEO_DOWNLOAD_INSTRUCTIONS}"
    )


class Molmo2SynCaptionsQA(DatasetBase):
    HF_SOURCE = "allenai/Molmo2-VideoCapQA"
    HF_SPLIT = "CapQA"
    LOCAL_NAME = "Molmo2-VideoCapQA"
    VIDEO_DIR = join(VIDEO_DATA_HOME, "youtube-cc") if VIDEO_DATA_HOME else None
    VIDEO_DOWNLOAD_INSTRUCTIONS = (
        "See the HF repo README for download instructions:\n"
        "  https://huggingface.co/datasets/allenai/Molmo2-VideoCapQA\n"
        "Place downloaded videos under MOLMO_DATA_DIR/video_datasets/molmo2-youtube-cc/ so the structure is:\n"
        "  molmo2-youtube-cc/youtube-cc-exist/{video_id}/{video_file}\n"
        "  molmo2-youtube-cc/youtube-cc-kw/{video_id}/{video_file}\n"
        "  molmo2-youtube-cc/youtube-cc-temporal/{video_id}/{video_file}"
    )

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    def __init__(self, split):
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        video2qas = defaultdict(list)
        for row in ds:
            video2qas[row["video_id"]].append(row)

        data = []
        for video_id, qas in video2qas.items():
            video_path = _find_video_by_id(
                self.VIDEO_DIR, video_id, self.VIDEO_DOWNLOAD_INSTRUCTIONS
            )
            msgs = []
            for qa in qas:
                answer = qa["Answer"]
                neg_options = list(qa["NegativeAnswers"])
                answer_idx = random.randint(0, len(neg_options))
                neg_options.insert(answer_idx, answer)
                msg = dict(
                    question=qa["Question"],
                    answer_idx=answer_idx,
                    options=neg_options,
                    style="video_multiple_choice",
                    category=qa.get("Category"),
                )
                msgs.append(msg)
            data.append({
                "video": video_path,
                "metadata": dict(example_id=f"molmo2_syn_qa_{video_id}"),
                "message_list": msgs,
            })
        return data

    def get(self, idx, rng):
        return self.data[idx]


def _get_video_point_preprocessed_path(local_name):
    """Get path for the preprocessed rows_with_clips pickle."""
    if VIDEO_DATA_HOME is None:
        return None
    return join(VIDEO_DATA_HOME, f"{local_name}_preprocessed.pkl")


class Molmo2VideoPoint(DatasetBase):
    """Loads allenai/Molmo2-VideoPoint from HuggingFace.

    Each example (msg) has: subset, example_id, label, answer, count,
    points (sorted), timestamps, question.
    Style is set dynamically in get(), not stored in load().
    """
    HF_SOURCE = "allenai/Molmo2-VideoPoint"
    HF_SPLIT = "train"
    HF_CONFIGS = [
        "action_or_event", "animal", "anomaly", "comparative reference",
        "indirect reference", "object", "referring expression", "spatial reference",
    ]
    LOCAL_NAME = "Molmo2-VideoPoint"
    VIDEO_DIR = VIDEO_DATA_HOME

    @classmethod
    def download(cls, n_procs=None):
        """Download HF configs, resolve video paths, and save preprocessed pickle."""
        all_rows = []
        for config in cls.HF_CONFIGS:
            config_ds = _load_hf_dataset(
                cls.HF_SOURCE, cls.HF_SPLIT,
                f"{cls.LOCAL_NAME}/{config}", config=config
            )
            all_rows.extend(config_ds)

        rows_with_clips = []
        for row in all_rows:
            # find video path based on video_id and file structures from source datasets
            # alternatively, define your own downloaded video's path based on video_id
            video_path = _find_video_point_path_by_id(
                cls.VIDEO_DIR, row["video_id"], row["video_source"]
            )
            rows_with_clips.append((dict(row), video_path, row["clip_start"], row["clip_end"]))

        preprocessed_path = _get_video_point_preprocessed_path(cls.LOCAL_NAME)
        if preprocessed_path:
            with open(preprocessed_path, "wb") as f:
                pickle.dump(rows_with_clips, f)
            logging.info(f"Saved {len(rows_with_clips)} preprocessed rows to {preprocessed_path}")

    def __init__(self,
        split: str,
        mode: str = "point_count",
        point_sort_by: str = "xy",
        max_seconds: int = None,
        multi_message_short_clips: bool = False,
        min_points: int = None,
        max_points: int = None,
        use_clips_from_metadata: bool = True,
        fps: int = 2,
        oversample: bool = False,
        p_multi_turn: float = 0.2
    ):
        self.mode = mode
        self.point_sort_by = point_sort_by
        self.max_seconds = max_seconds
        self.multi_message_short_clips = multi_message_short_clips
        self.min_points = min_points
        self.max_points = max_points
        self.use_clips_from_metadata = use_clips_from_metadata
        self.fps = fps
        self.oversample = oversample
        self.timestamp_step = 1.0 / self.fps
        self.p_multi_turn = p_multi_turn
        if self.use_clips_from_metadata:
            self.max_seconds = 63 # Clips from metadata are pre-computed for max 63 seconds
        if self.multi_message_short_clips:
            assert self.max_seconds > 0
        super().__init__(split)

    def load(self):
        preprocessed_path = _get_video_point_preprocessed_path(self.LOCAL_NAME)
        if not preprocessed_path or not exists(preprocessed_path):
            raise FileNotFoundError(
                f"Preprocessed data not found at {preprocessed_path}. "
                f"Run Molmo2VideoPoint.download() first."
            )
        logging.info(f"Loading preprocessed rows from {preprocessed_path}")
        with open(preprocessed_path, "rb") as f:
            rows_with_clips = pickle.load(f)

        # Group by (video_path, clip_start, clip_end)
        grouped = defaultdict(list)
        for row, video_path, clip_start, clip_end in rows_with_clips:
            key = (video_path, clip_start, clip_end)
            grouped[key].append(row)

        video2msgs = defaultdict(list)
        video_durations = {}
        data = []
        invalid_cnt = 0
        for (video_path, clip_start, clip_end), rows in grouped.items():
            if clip_start is not None and clip_end is not None:
                clip_duration = clip_end - clip_start
                if self.max_seconds is not None and clip_duration > self.max_seconds:
                    continue

            video_id = rows[0]["video_id"]
            video_duration = rows[0]["video_duration"]
            video_durations[video_path] = video_duration
            msgs = []
            for row in rows:
                label = row["label"]
                count = row["count"]
                if self.min_points is not None and count < self.min_points:
                    continue
                if self.max_points is not None and count > self.max_points:
                    continue
                timestamps = row["two_fps_timestamps"]
                points = row["points"]

                msg = {
                    "subset": row["category"],
                    "example_id": f"{video_id}_{label}",
                    "label": label,
                    "answer": str(count),
                    "count": count,
                    "points": points,
                    "timestamps": timestamps,
                    "question": row["question"],
                    "video_source": row["video_source"],
                    "video_duration": video_duration,
                    "annotator_unsure": row["annotator_unsure"],
                    "metadata": {
                        "points": points,
                        "timestamps": timestamps,
                        "subset": row["category"],
                        "count": count,
                        "video_id": video_id,
                    },
                }
                if timestamps:
                    ann_start = min(timestamps)
                    ann_end = max(timestamps)
                else:
                    # for zero count, use the entire video duration
                    ann_start = 0.0
                    ann_end = min(video_duration, self.max_seconds) if self.max_seconds is not None else video_duration
                if self.max_seconds is not None:
                    if video_duration <= self.max_seconds:
                        msg["clip_start_time"] = 0.0
                        msg["clip_end_time"] = video_duration
                    elif clip_start is not None and clip_end is not None:
                        msg["clip_start_time"] = clip_start
                        msg["clip_end_time"] = clip_end
                    else:
                        if self.use_clips_from_metadata:
                            # Only use examples with pre-computed clip boundaries
                            continue
                        # sample a random clip covering annotated timestamps
                        try:
                            timestamp_step = 1.0 / self.fps
                            rand_start, rand_end = sample_random_clip(
                                video_duration=video_duration,
                                start_time=ann_start,
                                end_time=ann_end,
                                min_seconds=timestamp_step,
                                max_seconds=self.max_seconds,
                                timestamp_step=timestamp_step,
                                seed=42,
                            )
                        except ValueError:
                            continue
                        msg["clip_start_time"] = rand_start
                        msg["clip_end_time"] = rand_end
                    try:
                        assert msg['clip_start_time'] >= 0, msg['clip_start_time']
                        assert msg['clip_end_time'] <= video_duration, (msg['clip_end_time'], video_duration)
                        assert msg['clip_start_time'] < msg['clip_end_time'], (msg['clip_start_time'], msg['clip_end_time'])
                        assert msg['clip_end_time'] - msg['clip_start_time'] <= self.max_seconds, (msg['clip_end_time'], msg['clip_start_time'], self.max_seconds)
                        assert msg['clip_start_time'] <= ann_start, (msg['clip_start_time'], ann_start)
                        assert msg['clip_end_time'] >= ann_end, (msg['clip_end_time'], ann_end)
                    except AssertionError:
                        invalid_cnt += 1
                        continue
                    
                all_sorted_points = []
                all_timestamps = []
                for i, ts in enumerate(msg["timestamps"]):
                    frame_points = msg["points"][i]
                    # assuming we use 2fps annotations by default
                    assert ts / 0.5 == int(
                        ts / 0.5
                    ), f"Original timestamp {ts} not aligned to 0.5s intervals"

                    if self.point_sort_by == "xy":
                        sorted_points = sorted(frame_points, key=lambda p: (p["x"], p["y"]))
                    elif self.point_sort_by == "yx":
                        sorted_points = sorted(frame_points, key=lambda p: (p["y"], p["x"]))
                    else:
                        sorted_points = frame_points
                    all_sorted_points.append(sorted_points)
                    if "clip_start_time" in msg:
                        ts = ts - msg["clip_start_time"]
                    # align timestamps to the specified fps intervals
                    ts = math.floor(ts / 0.5) * self.timestamp_step
                    all_timestamps.append(ts)

                msg["points"] = all_sorted_points
                msg["timestamps"] = all_timestamps

                if self.oversample:
                    n_points = sum(len(x) for x in all_sorted_points)
                    if n_points <= 5:
                        oversample = 1
                    elif n_points <= 25:
                        oversample = 2
                    else:
                        oversample = 4
                    for k in range(oversample):
                        video2msgs[(video_path, k)].append(msg)
                else:
                    video2msgs[(video_path, 0)].append(msg)

        if self.multi_message_short_clips:
            n_multi_turn = 0
            for (video_path, _), msgs in video2msgs.items():
                duration = video_durations[video_path]
                if duration <= self.max_seconds and len(msgs) > 1:
                    # Normalize clip times: for short videos, all messages should
                    # share the same clip window (0, duration). They can differ slightly
                    # when the same video appears in multiple groups with different
                    # video_duration values due to float precision or data inconsistency.
                    start, end = 0.0, duration
                    n_multi_turn += 1
                    data.append(dict(
                        message_list=msgs,
                        video=video_path,
                        metadata=dict(
                            clip_start_time=start,
                            clip_end_time=end
                        )
                    ))
                else:
                    for msg in msgs:
                        if "clip_start_time" in msg:
                            msg["metadata"] = dict(
                                clip_start_time=msg["clip_start_time"],
                                clip_end_time=msg["clip_end_time"]
                            )
                        msg["video"] = video_path
                        data.append(msg)
            logging.info(f"Have {n_multi_turn} multi-turn video pointing messages")
        elif self.max_seconds > 0:
            for (video_path, _), msgs in video2msgs.items():
                for msg in msgs:
                    if "clip_start_time" in msg:
                        msg["metadata"] = dict(
                            clip_start_time=msg["clip_start_time"],
                            clip_end_time=msg["clip_end_time"]
                        )
                    msg["video"] = video_path
                    data.append(msg)
        else:
            for (video_path, _), msgs in video2msgs.items():
                data.append({
                    "video": video_path,
                    "message_list": msgs,
                })
        if invalid_cnt > 0:
            logging.warning(f"Skipped {invalid_cnt} examples due to invalid clip times")
        return data

    def _get_style(self, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return f"video_{style}"

    def get(self, idx, rng):
        example = dict(self.data[idx])
        if "message_list" in example and len(example["message_list"]) > 1:
            with_style = []
            for message in example.pop("message_list"):
                style = self._get_style(rng)
                if style == "video_point" and "question" in message:
                    # message["question"] is gpt-generated counting question e.g. "how many"
                    # for video_point only without counting, we template the question in data_formatter.py instead
                    del message["question"]
                with_style.append(dict(message, style=style))
            assert len(with_style) > 0
            if rng.random() < self.p_multi_turn:
                example["multi_turn_messages"] = with_style
            else:
                example["message_list"] = with_style
            return example
        else:
            style = self._get_style(rng)
            if style == "video_point" and "question" in example:
                del example["question"]
            return set_example_style(example, style)


class Molmo2VideoPointEval(DatasetBase):
    """Loads allenai/Molmo2-VideoPointEval from HuggingFace.

    Each example has: label, question, style, timestamps, points (normalized 0-100),
    metadata with gt_abs_triplets, gt_abs_masks, video_duration, video_height, video_width.
    """
    HF_SOURCE = "allenai/Molmo2-VideoPointEval"
    HF_SPLIT = "val"
    LOCAL_NAME = "Molmo2-VideoPointEval"
    VIDEO_DIR = VIDEO_DATA_HOME

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    def __init__(self, split):
        assert split in ["val"]
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        # First pass: resolve video paths
        rows_with_paths = []
        for row in ds:
            video_id = row["video_id"]
            video_path = _find_video_point_path_by_id(
                self.VIDEO_DIR, video_id, row.get("video_source")
            )
            rows_with_paths.append((row, video_path))

        # Group by video_path
        grouped = defaultdict(list)
        for row, video_path in rows_with_paths:
            grouped[video_path].append(row)

        data = []
        for video_path, rows in grouped.items():
            for row in rows:
                height = row["height"]
                width = row["width"]
                video_duration = row["video_duration"]
                timestamps = row["two_fps_timestamps"]
                raw_points = row["points"]
                raw_masks = row["masks"]

                # Build gt_abs_triplets: (timestamp, x_pixel, y_pixel)
                gt_abs_triplets = []
                for i, pt in enumerate(raw_points):
                    t = timestamps[i]
                    gt_abs_triplets.append((t, pt["x"], pt["y"]))

                # Build gt_abs_masks: list of {time -> decoded_mask_array}
                gt_abs_masks = []
                for frame_masks in raw_masks:
                    time2mask = {}
                    for mask in frame_masks:
                        # frame_id / 2.0 assumes 2fps annotation
                        mask_t = mask["frame_id"] / 2.0
                        time2mask[mask_t] = _decode_rle_mask(mask["rle"])
                    gt_abs_masks.append(time2mask)

                # Normalize points from pixel coords to 0-100 range
                # Each point becomes a single-element list
                norm_points = [
                    [{"x": round(pt["x"] / width * 100, 1), "y": round(pt["y"] / height * 100, 1)}]
                    for pt in raw_points
                ]

                label = row["label"]
                data.append({
                    "video": video_path,
                    "label": label,
                    "question": f"How many \"{label}\" are there in the video?",
                    "style": "video_point_count",
                    "timestamps": timestamps,
                    "points": norm_points,
                    "metadata": {
                        "gt_abs_triplets": gt_abs_triplets,
                        "gt_abs_masks": gt_abs_masks,
                        "video_duration": video_duration,
                        "video_height": height,
                        "video_width": width,
                    },
                })
        return data

    def get(self, idx, rng):
        return self.data[idx]


class Molmo2VideoCountEval(DatasetBase):
    """Loads allenai/Molmo2-VideoCountEval from HuggingFace.

    Each example has: label, question, answer, count, style, subset,
    metadata with clip_start_time, clip_end_time, video_duration.
    """
    HF_SOURCE = "allenai/Molmo2-VideoCountEval"
    HF_SPLIT = "val"
    LOCAL_NAME = "Molmo2-VideoCountEval"
    VIDEO_DIR = VIDEO_DATA_HOME

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    def __init__(self, split):
        assert split in ["val"]
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        data = []
        for row in ds:
            video_id = row["video_id"]
            video_source = row["video_source"]
            video_path = _find_video_point_path_by_id(
                self.VIDEO_DIR, video_id, video_source
            )

            clip_start = row["clip_start"]
            clip_end = row["clip_end"]
            count = row["count"]
            label = row["label"]

            data.append({
                "video": video_path,
                "video_id": video_id,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "subset": row["category"],
                "example_id": f"{video_id}_{label}",
                "label": label,
                "question": row["question"],
                "answer": str(count),
                "count": count,
                "style": "video_count",
                "video_source": row["video_source"],
                "video_duration": row["video_duration"],
                "metadata": {
                    "video_id": video_id,
                    "subset": row["category"],
                    "count": count,
                    "video_duration": row["video_duration"],
                    "clip_start_time": clip_start,
                    "clip_end_time": clip_end,
                },
            })
        return data

    def get(self, idx, rng):
        return self.data[idx]


class Molmo2HumanQA(DatasetBase):
    HF_SOURCE = "allenai/Molmo2-AskModelAnything"
    HF_SPLIT = "HumanQA"
    LOCAL_NAME = "Molmo2-AskModelAnything"
    VIDEO_DIR = join(VIDEO_DATA_HOME, "youtube-cc") if VIDEO_DATA_HOME else None
    VIDEO_DOWNLOAD_INSTRUCTIONS = (
        "See the HF repo README for download instructions:\n"
        "  https://huggingface.co/datasets/allenai/Molmo2-AskModelAnything\n"
        "Place downloaded videos under MOLMO_DATA_DIR/video_datasets/youtube-cc/ so the structure is:\n"
        "  youtube-cc/youtube-cc-exist/{video_id}/{video_file}\n"
        "  youtube-cc/youtube-cc-kw/{video_id}/{video_file}\n"
        "  youtube-cc/youtube-cc-temporal/{video_id}/{video_file}"
    )

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    def __init__(self, split):
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        video2qas = defaultdict(list)
        for row in ds:
            q = row["question"].strip()
            a = row["answer"].strip()
            if a:
                video2qas[row["video_id"]].append(dict(question=q, answer=a, style="user_qa"))

        data = []
        for video_id, msgs in video2qas.items():
            video_path = _find_video_by_id(
                self.VIDEO_DIR, video_id, self.VIDEO_DOWNLOAD_INSTRUCTIONS
            )
            data.append({
                "video": video_path,
                "message_list": msgs,
            })
        return data

    def get(self, idx, rng):
        return self.data[idx]


class Molmo2SynCaptionsSubtitleQA(DatasetBase):
    HF_SOURCE = "allenai/Molmo2-VideoSubtitleQA"
    HF_SPLIT = "SubtitleQA"
    LOCAL_NAME = "Molmo2-VideoSubtitleQA"
    VIDEO_DIR = join(VIDEO_DATA_HOME, "youtube-cc") if VIDEO_DATA_HOME else None
    VIDEO_DOWNLOAD_INSTRUCTIONS = (
        "See the HF repo README for download instructions:\n"
        "  https://huggingface.co/datasets/allenai/Molmo2-VideoSubtitleQA\n"
        "Place downloaded videos under MOLMO_DATA_DIR/video_datasets/youtube-cc/ so the structure is:\n"
        "  youtube-cc/youtube-cc-exist/{video_id}/{video_file}\n"
        "  youtube-cc/youtube-cc-kw/{video_id}/{video_file}\n"
        "  youtube-cc/youtube-cc-temporal/{video_id}/{video_file}"
    )

    @classmethod
    def download(cls, n_procs=None):
        _load_hf_dataset(cls.HF_SOURCE, cls.HF_SPLIT, cls.LOCAL_NAME)

    def __init__(self, split):
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.HF_SPLIT, self.LOCAL_NAME)

        video2rows = defaultdict(list)
        for row in ds:
            video2rows[row["video_id"]].append(row)

        data = []
        for video_id, rows in video2rows.items():
            video_path = _find_video_by_id(
                self.VIDEO_DIR, video_id, self.VIDEO_DOWNLOAD_INSTRUCTIONS
            )
            msgs = []
            for row in rows:
                answer = row["Answer"]
                neg_options = list(row["NegativeAnswers"])
                answer_idx = random.randint(0, len(neg_options))
                neg_options.insert(answer_idx, answer)
                msg = dict(
                    question=row["Question"],
                    answer_idx=answer_idx,
                    options=neg_options,
                    style="video_multiple_choice",
                    category=row.get("Category"),
                )
                msgs.append(msg)

            # Build subtitle dict: (start, end) -> text
            # All rows for the same video share the same subtitle
            subtitle = {}
            for sub in rows[0]["subtitle"]:
                subtitle[(sub["start"], sub["end"])] = sub["text"]

            data.append({
                "video": video_path,
                "metadata": dict(example_id=f"molmo2_syn_qa_{video_id}"),
                "subtitle": subtitle,
                "message_list": msgs,
            })
        return data

    def get(self, idx, rng):
        return self.data[idx]


class Molmo2SyntheticPoint(Dataset):

    @classmethod
    def download(cls, n_procs=None):
        (datasets.load_dataset_builder("allenai/MolmoPoint-Synthetic")
         .download_and_prepare(num_proc=n_procs))

    def __init__(self, split, keep_in_memory=False,
                 p_intent=0.8, use_name_as_label=False):
        all_parts = []
        for part in ["benchmark", "desktop", "mobile", "web"]:
            all_parts.append(datasets.load_dataset("allenai/MolmoPoint-Synthetic", part, split=split, keep_in_memory=keep_in_memory))
        self.data = datasets.concatenate_datasets(all_parts)
        self.p_intent = p_intent
        self.use_name_as_label = use_name_as_label

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        message_list = []
        for annotation in ex["annotation"]:
            msg = dict(
                points=np.array([annotation["x_center"], annotation["y_center"]])[None, :],
                style="pointing"
            )
            name = annotation["name"]
            if len(annotation["intent"]) > 0 and (not name or rng.random() < self.p_intent):
                if name and self.use_name_as_label:
                    msg["label_cased"] = annotation["name"]
                msg["question"] = rng.choice(annotation["intent"])
            elif name:
                # Use the point template
                msg["label_cased"] = annotation["name"]
            else:
                continue
            message_list.append(msg)
        assert len(message_list) > 0
        return dict(
            image=ex["image"],
            message_list=message_list,
            metadata=dict(example_id=ex["id"])
        )

class Molmo2Captions(DatasetBase):
    """Loads allenai/Molmo2-Cap from HuggingFace.

    A video captioning dataset with detailed captions (~900 words avg).
    Each example has: video_id (YouTube), video_start/end timestamps,
    merged_caption, video_frame_merged_caption, annotation_score, and
    per-clip/per-frame captions and transcripts.

    Videos must be downloaded separately (YouTube).

    Mirrors the behavior of VixMoCaptions from VideoOlmo, with caption types
    mapped as follows:
        HF column                    -> style
        video_caption                -> video_short_caption
        merged_caption               -> video_merged_caption
        video_frame_merged_caption   -> video_long_caption
    """
    HF_SOURCE = "allenai/Molmo2-Cap"
    LOCAL_NAME = "Molmo2-Cap"
    VIDEO_DIR = VIDEO_DATA_HOME

    CORRUPT_VIDEO_IDS = {
        "0OfUqrbRa2Y",
        "mTXBQzptWAg",
        "NyuA4FatDQk",
        "a_",
        "gsfIHiBB6xE",
        "bXApJtAf6Qo",
        "N8BlpYSpgg4",
    }

    # Maps HF column name -> style (same mapping as VixMoCaptions.version2style)
    CAPTION_KEY_TO_STYLE = {
        "video_caption": "video_short_caption",
        "merged_caption": "video_merged_caption",
        "video_frame_merged_caption": "video_long_caption",
    }

    @classmethod
    def download(cls, n_procs=None):
        for split in ["train", "val"]:
            _load_hf_dataset(cls.HF_SOURCE, split, f"{cls.LOCAL_NAME}/{split}")

    def __init__(
        self,
        split,
        include_video_caption=False,
        include_merged_caption=False,
        include_video_frame_merged_caption=False,
        min_score=0,
        max_caption_per_video=4,
    ):
        assert any([
            include_video_caption,
            include_merged_caption,
            include_video_frame_merged_caption,
        ]), "At least one caption type must be included"
        self.include_video_caption = include_video_caption
        self.include_merged_caption = include_merged_caption
        self.include_video_frame_merged_caption = include_video_frame_merged_caption
        self.min_score = min_score
        self.max_caption_per_video = max_caption_per_video
        super().__init__(split)

    def load(self):
        ds = _load_hf_dataset(self.HF_SOURCE, self.split, f"{self.LOCAL_NAME}/{self.split}")

        cap_versions = []
        if self.include_merged_caption:
            cap_versions.append("merged_caption")
        if self.include_video_caption:
            cap_versions.append("video_caption")
        if self.include_video_frame_merged_caption:
            cap_versions.append("video_frame_merged_caption")

        data = []
        for row in ds:
            if row["video_id"] in self.CORRUPT_VIDEO_IDS:
                continue
            if row["annotation_score"] < self.min_score:
                continue

            messages = []
            for version in cap_versions:
                caption = row[version]
                if not caption:
                    continue
                messages.append(dict(
                    text=caption,
                    style=self.CAPTION_KEY_TO_STYLE[version],
                ))

            if len(messages) == 0:
                continue

            for msg_group in split_into_groups(messages, self.max_caption_per_video):
                data.append((msg_group, row))

        return data

    def get(self, idx, rng):
        messages, row = self.data[idx]
        video_id = row["video_id"]
        # find video path based on video_id and file structures from source datasets
        # alternatively, define your own downloaded video's path based on video_id
        video_path = _find_video_cap_path_by_id(self.VIDEO_DIR, video_id)
        return {
            "video": video_path,
            "metadata": dict(
                video_path=video_path,
                example_id=idx,
                video_id=video_id,
                split=self.split,
            ),
            "message_list": messages,
        }
