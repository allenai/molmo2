import logging
import os
import json
import shutil
import subprocess

from os.path import join, exists
import multiprocessing as mp
from glob import glob

import datasets

from olmo.io import (
    read_file,
    file_exists
)
from olmo.torch_util import get_global_rank

from olmo.data.dataset import VIDEO_DATA_HOME
from olmo.data.utils import maybe_download_and_unzip, maybe_download_file
from olmo.data.academic_video_track_datasets import (
    _encode_frames_to_video_worker,
    _extract_frames_worker,
    _load_hf_dataset,
    TrackingDataset, PointTrack
)

from huggingface_hub import snapshot_download, hf_hub_download
import zipfile

from tqdm import tqdm

from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

VIDEO_TRACK_SOURCES = [
    'MOSE',          # https://huggingface.co/datasets/FudanCVL/MOSE
    'MOSEv2',        # https://huggingface.co/datasets/FudanCVL/MOSEv2
    'SAV',           # https://ai.meta.com/datasets/segment-anything-video/
    'VIPSeg',        # https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/
    'AnimalTrack',   # https://hengfan2010.github.io/projects/AnimalTrack/
    'APTv2',         # https://github.com/ViTAE-Transformer/APTv2
    'BFT',           # https://george-zhuang.github.io/nettrack/
    'SoccerNet',     # https://www.soccer-net.org/data
    'SportsMOT',     # https://codalab.lisn.upsaclay.fr/competitions/12424#participate
    'TeamTrack',     # https://github.com/AtomScott/TeamTrack
    'MOT20',         # https://motchallenge.net/data/MOT20/
    'PersonPath22',  # https://amazon-science.github.io/tracking-dataset/personpath22.html
    'DanceTrack',    # https://github.com/DanceTrack/DanceTrack
    'BDD100K',       # http://128.32.162.150/bdd100k/video_parts/
    'UAVDT',         # https://sites.google.com/view/grli-uavdt/
    'SeaDronesSee',  # https://seadronessee.cs.uni-tuebingen.de/dataset
]

VIDEO_TRACK_DATA_HOME = join(VIDEO_DATA_HOME, "video_track")

def split_video_ffmpeg(input_video, output_video, start_sec, duration_sec):
    """Split video using ffmpeg with re-encoding for reliability.

    Why re-encode instead of copy?
    - Codec copy (-c copy) cuts at non-keyframe positions
    - This causes videos to appear stuck or end early in browsers
    - Re-encoding ensures clean cuts with proper keyframes

    Args:
        input_video: Path to input video
        output_video: Path to output video
        start_sec: Start time in seconds
        duration_sec: Duration of clip in seconds
    """
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    cmd = [
        'ffmpeg',
        '-ss', str(start_sec),
        '-i', input_video,
        '-t', str(duration_sec),
        '-c:v', 'libopenh264',
        '-b:v', '4M',
        '-pix_fmt', 'yuv420p',
        '-g', '60',
        '-movflags', '+faststart',
        '-y',
        output_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")


def _split_video_worker(item):
    """Multiprocessing worker: copy or trim a full video into a clip.

    If 'is_same_video' is set, the full video is copied (clip == full video).
    Otherwise, the video is trimmed using ffmpeg (clip is a subset).
    """
    source_video = item['source_video']
    output_path = item['output_path']

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if item.get('is_same_video'):
            shutil.copy2(source_video, output_path)
            return output_path, True, None

        fps = item['fps']
        start_sec = item['start_frame'] / fps
        duration_sec = (item['end_frame'] - item['start_frame'] + 1) / fps
        split_video_ffmpeg(source_video, output_path, start_sec, duration_sec)
        return output_path, True, None
    except Exception as e:
        return output_path, False, str(e)


class Molmo2VideoTrackInstruction(TrackingDataset):
    """Training dataset for video object tracking with pre-computed instruction data.

    This is the primary class used for training. It loads from the instruction-formatted
    HF dataset which contains pre-computed frame trajectories at specific sampling_fps values,
    matching the exact data distribution used to train the released model.

    Schema matches academic track format (frame_trajectories, expression, width, height),
    so it inherits load(), _create_message_list(), _get_style() from TrackingDataset.
    Only overrides get() to resolve video paths per-source via Molmo2TrackVideoSource classes.

    Videos are downloaded via Molmo2VideoTrack.download(), which handles the per-source
    frame extraction and video encoding pipeline.

    Related classes:
        - Molmo2VideoTrack: raw annotations dataset, used for video download only
        - Molmo2VideoTrackEval: evaluation dataset with ground-truth masks

    Source: https://huggingface.co/datasets/allenai/molmo2-track-instruction
    """
    HF_SOURCE = "allenai/molmo2-track-instruction"
    DATASET_NAME = "molmo2-track-instruction"
    LOCAL_NAME = "molmo2-track-instruction"
    TASKS = ["track" , "ground", "single_point_track"]
    SPLIT_MAP = {
        "train": "train",
    }
    SOURCES =[
        'mose',
        'mosev2',
        'sav',
        'vipseg',
        'animaltrack',
        'APTv2',
        'bft',
        'soccernet',
        'sportsmot',
        'teamtrack',
        'mot2020',
        'personpath22',
        'dancetrack',
        'bdd100k',
        'uavdt',
        'seadrones',
    ]

    @classmethod
    def _get_video_dir_for_source(cls, source):
        video_source_downloader = get_video_source([source])[source]
        return video_source_downloader._get_video_dir('train')

    @classmethod
    def download(cls, n_procs=1, sources=None):
        """Download HF annotations and create videos for each source.

        Args:
            sources: list of source names to download, or None for all.
            n_procs: num workers for parallel video creation.
        """

        # Build {video_source: {clip_name: {video, fps, start_frame, end_frame}}}
        source_clip_map = {}
        data_split = 'train'
        for task in cls.TASKS:
            local_dir = cls._get_local_dir(task, data_split)
            ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir, config=task, overwrite_cache=True)
            for row in ds.select_columns(["video", "clip", "fps", "video_dataset", "start_frame", "end_frame"]):
                src = row['video_dataset']
                if src not in source_clip_map:
                    source_clip_map[src] = {}
                source_clip_map[src][row['clip']] = dict(
                    video=row['video'],
                    fps=row['fps'],
                    start_frame=row['start_frame'],
                    end_frame=row['end_frame'],
                )

        # Download + create videos per source
        for source, source_cls in get_video_source(sources).items():
            clip_map = source_clip_map.get(source, {})
            log.info(f"Processing source '{source}' ({len(clip_map)} clips)...")
            source_cls.download(clip_map, n_procs, split='train')

    def _get_video_path(self, video_id):
        """Not used — overridden by get() which resolves per-source video dirs."""
        raise NotImplementedError("Use get() directly; videos are resolved per-source.")

    def get(self, idx, rng):
        ex = self.data[idx]
        video_fps = ex['fps']

        # Clips live in videos/ dir for each source
        source = ex['video_dataset']
        video_dir = self._get_video_dir_for_source(source)
        video_path = join(video_dir, ex['clip'] + '.mp4')

        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['expression'],
            'w': ex['width'],
            'h': ex['height'],
            'video_fps': video_fps,
            'orig_video': ex['video'],
            'video': ex['clip'],
            'video_dataset': ex['video_dataset'],
        }

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': self.sampling_fps or ex['sampling_fps'],
            }

        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': ex['sampling_fps'],
            'metadata': metadata,
        }
class Molmo2VideoTrack(TrackingDataset):
    """Raw video object tracking annotations dataset.

    Contains 29.7k examples across 16 source datasets with raw per-object point
    annotations, clip boundaries (start_frame/end_frame), and source metadata.

    NOT used for training — use Molmo2VideoTrackInstruction instead, which loads
    the instruction-formatted dataset with pre-computed trajectories matching the
    exact training distribution. This class exists for:
        1. Video download: download() handles per-source frame extraction and video encoding
        2. Dataset exploration: loading the raw annotations for inspection or re-processing

    Source: https://huggingface.co/datasets/allenai/Molmo2-VideoTrack
    """
    HF_SOURCE = "allenai/Molmo2-VideoTrack"
    LOCAL_NAME = "Molmo2-VideoTrack"
    # TASKS = ["track", "ground", "single_point_track"]
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
    }
    SOURCES = [
        'mose',
        'mosev2',
        'sav',
        'vipseg',
        'animaltrack',
        'APTv2',
        'bft',
        'soccernet',
        'sportsmot',
        'teamtrack',
        'mot2020',
        'personpath22',
        'dancetrack',
        'bdd100k',
        'uavdt',
        'seadrones',
    ]

    @classmethod
    def _get_video_dir_for_source(cls, source):
        video_source_downloader = get_video_source([source])[source]
        return video_source_downloader._get_video_dir('train')

    @classmethod
    def download(cls, n_procs=1, sources=None):
        """Download HF annotations and create videos for each source.

        Args:
            sources: list of source names to download, or None for all.
            n_procs: num workers for parallel video creation.
        """
        ds = _load_hf_dataset(cls.HF_SOURCE, 'train', local_name=cls.LOCAL_NAME, overwrite_cache=True)

        # Build {video_source: {clip_name: {video, fps, start_frame, end_frame}}}
        source_clip_map = {}
        for row in ds.select_columns(["video", "clip", "fps", "video_dataset", "start_frame", "end_frame"]):
            src = row['video_dataset']
            if src not in source_clip_map:
                source_clip_map[src] = {}
            source_clip_map[src][row['clip']] = dict(
                video=row['video'],
                fps=row['fps'],
                start_frame=row['start_frame'],
                end_frame=row['end_frame'],
            )

        # Download + create videos per source
        for source, source_cls in get_video_source(sources).items():
            clip_map = source_clip_map.get(source, {})
            log.info(f"Processing source '{source}' ({len(clip_map)} clips)...")
            source_cls.download(clip_map, n_procs, split='train')

    def __init__(self, split, task, sampling_fps, use_fps_sampling=True):
        assert task in self.TASKS, f"Invalid task: {task}. Available: {self.TASKS}"
        self.split = split
        self.task = task
        self.sampling_fps = sampling_fps
        self.use_fps_sampling = use_fps_sampling
        self.data_split = self.SPLIT_MAP[split]
        self.data_lookup = {} # example_id -> index for get_by_example_id
        self.data = self.load()

    def load(self):
        data = _load_hf_dataset(self.HF_SOURCE, 'train', local_name=self.LOCAL_NAME)
        self.data_lookup = {ex_id: i for i, ex_id in enumerate(data["id"])}

        # Filter out examples where video fps is not divisible by sampling_fps
        n_pre = len(data)
        data = data.filter(lambda fps: fps % self.sampling_fps == 0, input_columns="fps")
        if n_pre != len(data):
            log.info(f"Filtered {n_pre - len(data)}/{n_pre} examples: fps not divisible by sampling_fps={self.sampling_fps}")

        return data

    def __len__(self):
        return len(self.data)

    def _create_message_list(self, ex):
        """Convert HF row to message_list format matching TrackingDataset."""
        style = self._get_style()

        # Build point trajectories from row data
        point_tracks: List[PointTrack] = []
        points_data = ex['points']  # list [{'object_id': 'id', 'points': [[x,y],...]}]
        fps = ex['fps']
        mask_ids = ex['mask_id']

        # Map mask_id to sequential index
        object_id_to_idx = {mid: idx for idx, mid in enumerate(mask_ids)}

        # Gather points by frame
        points_per_frame: Dict[int, Dict[int, Dict[str, Any]]] = {} # frame_idx -> object_idx -> Point
        for points_per_object in points_data:
            obj_id = points_per_object['object_id']
            obj_idx = object_id_to_idx[obj_id]
            points = points_per_object['points']
            for frame_idx, point in enumerate(points):
                if point:
                    point_info = {
                        'point': point,
                        'occluded': False,  # HF dataset doesn't specify occlusion, default to False
                    }
                    if frame_idx not in points_per_frame:
                        points_per_frame[frame_idx] = {}
                    points_per_frame[frame_idx][obj_idx] = point_info

        # Convert points_per_frame to point_tracks list
        for frame_idx, objects in points_per_frame.items():
            time = frame_idx / fps
            if frame_idx % (fps // self.sampling_fps) == 0:  # sample according to sampling_fps
                point_tracks.append({
                    'frame': frame_idx,
                    'time': time,
                    'points': objects
                })

        # Sort by order of occurrence
        point_tracks.sort(key=lambda x: x['frame'])
        if not point_tracks:
            point_tracks = None

        return [{
            "style": style,
            "question": ex['exp'],
            "label": ex['exp'],
            "points": point_tracks,
            "sampling_fps": self.sampling_fps,
            "width": ex['w'],
            "height": ex['h'],
        }]

    def get(self, idx, rng):
        ex = self.data[idx]
        video_fps = ex['fps']

        # Videos come from different source datasets, stored under VIDEO_TRACK_DATA_HOME
        source = ex['video_dataset']
        video_dir = self._get_video_dir_for_source(source)
        video_rel_path = ex['clip'] + '.mp4'
        video_path = join(video_dir, video_rel_path)

        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['exp'],
            'w': ex['w'],
            'h': ex['h'],
            'video_fps': video_fps,
            'orig_video': ex['video'],
            'video': ex['clip'],
            'video_dataset': ex['video_dataset'],
        }

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': self.sampling_fps,
            }

        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': self.sampling_fps,
            'metadata': metadata,
        }

    def get_by_example_id(self, example_id):
        idx = self.data_lookup.get(example_id)
        if idx is not None:
            return self.get(idx, None)
        log.warning(f"Example ID '{example_id}' not found.")
        return None

class Molmo2VideoTrackEval(Molmo2VideoTrack):
    """Evaluation dataset for video object tracking.

    Contains 3.7k examples across 5 source datasets with ground-truth masks
    for computing segmentation metrics (J&F). Extends Molmo2VideoTrack (raw format)
    because eval requires per-frame masks which are not in the instruction dataset.
    """
    HF_SOURCE = "allenai/Molmo2-VideoTrackEval"
    LOCAL_NAME = "Molmo2-VideoTrackEval"
    TASKS = ["track", ]
    SPLIT_MAP = {
        "test": "test",
    }

    # config -> video sources for downloads
    CONFIGS = {
        'animal': ['APTv2'],
        'dance': ['dancetrack'],
        'sports': ['sportsmot'],
        'person': ['personpath22'],
        'misc': ['sav']
    }

    @classmethod
    def _get_video_dir_for_source(cls, source, split='test'):
        video_source_downloader = get_video_source([source])[source]
        return video_source_downloader._get_video_dir(split)

    @classmethod
    def download(cls, n_procs=1, configs=None, sources=None):
        """Download HF annotations and create videos for each source in eval configs.

        Args:
            n_procs: num workers for parallel video creation.
            configs: list of config names to download, or None for all.
            sources: list of source names to download, or None (derived from configs).
        """
        ds = _load_hf_dataset(cls.HF_SOURCE, 'test', local_name=cls.LOCAL_NAME, overwrite_cache=True)

        if configs is None:
            configs = list(cls.CONFIGS.keys())
        log.info(f"[{cls.__name__}] Downloading eval configs: {configs}")

        # Build {video_source: {video_relpath: fps}} from HF data
        source_clip_map = {}
        for row in ds.select_columns(["video", "clip", "fps", "video_dataset", "start_frame", "end_frame"]):
            src = row['video_dataset']
            if src not in source_clip_map:
                source_clip_map[src] = {}

            clip_name = row["clip"]
            source_clip_map[src][clip_name] = dict(
                video=row['video'],
                fps=row['fps'],
                start_frame=row['start_frame'],
                end_frame=row['end_frame'],
            )

        # Resolve configs -> sources if sources not explicitly given
        if sources is None:
            sources = []
            for config in configs:
                if config not in cls.CONFIGS:
                    log.warning(f"Unknown config '{config}'. Available: {list(cls.CONFIGS.keys())}")
                    continue
                sources.extend(cls.CONFIGS[config])

        # Download + create videos per source
        for source, source_cls in get_video_source(sources).items():
            clip_map = source_clip_map.get(source, {})
            log.info(f"Processing source '{source}' ({len(clip_map)} clips)...")
            source_cls.download(clip_map, n_procs, split='test')

    def __init__(self, split, task, sampling_fps, use_fps_sampling=True, configs=None): # require sampling_fps for eval
        assert task in self.TASKS, f"Invalid task: {task}. Available: {self.TASKS}"
        self.split = split
        self.task = task
        self.sampling_fps = sampling_fps
        self.use_fps_sampling = use_fps_sampling
        self.data_split = self.SPLIT_MAP[split]
        self.data_lookup = {} # example_id -> index for get_by_example_id
        self.configs = configs
        self.data = self.load()

    def load(self):
        data = _load_hf_dataset(self.HF_SOURCE, 'test', local_name=self.LOCAL_NAME)

        if self.configs is not None:
            log.info(f"Filtering dataset to configs: {self.configs}")
            sources = set()
            for config in self.configs:
                sources.update(self.CONFIGS[config])
            pre = len(data)
            data = data.filter(lambda src: src in sources, input_columns="video_dataset")
            log.info(f"Filtered to sources {sources} for configs {self.configs}: {len(data)}/{pre} examples remaining")

        self.data_lookup = {ex_id: i for i, ex_id in enumerate(data["id"])}

        return data

    def get(self, idx, rng):
        ex = self.data[idx]
        video_fps = ex['fps']

        # Clips live in videos/ dir for each source (symlinked or trimmed from full_videos/)
        source = ex['video_dataset']
        video_dir = self._get_video_dir_for_source(source)
        video_rel_path = ex['clip'] + '.mp4'
        video_path = join(video_dir, video_rel_path)

        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['exp'],
            'w': ex['w'],
            'h': ex['h'],
            'video_fps': video_fps,
            'orig_video': ex['video'],
            'video': ex['clip'],
            'video_dataset': ex['video_dataset'],
        }

        # Preprocess segmentation masks to {'object_id': [masks]} format for eval
        masks = ex['masks'] # [{'masks': [masks], 'object_id': mask_id}]
        for idx, mask in enumerate(masks):
            mask['frame'] = idx
        mask_id = ex['mask_id']
        assert len(masks) == len(mask_id), f"Mask count {len(masks)} does not match mask_id count {len(mask_id)}"
        masks_for_eval = {m['object_id']: m['masks'] for m in masks}
        metadata.update({
            'masks': masks_for_eval,
            'mask_id': mask_id,
            'points': message_list[0]['points'],
        })

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': self.sampling_fps,
            }

        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': self.sampling_fps,
            'metadata': metadata,
        }


class Molmo2TrackVideoSource:
    """Base class for per-source video downloaders for Molmo2-VideoTrack
    and video path resolution for loading.

    Each subclass handles downloading raw frames and creating .mp4 videos
    for one of the 16 video sources used in Molmo2-VideoTrack.

    Download pipeline (subclasses override _get_frames_dir and _prepare_annotation_dir):
      1. _check_videos()             -> return early if all clips present in videos/
      2. _prepare_annotation_dir()   -> download + extract raw frames
      3. _build_video_work_items()   -> (full_video_items, clip_items)
      4. _create_videos()            -> frames -> full_videos/
      5. _create_clips()             -> full_videos/ -> videos/ (symlink or trim)
      6. _check_videos()             -> verify
    """
    SOURCE_NAME = None   # matches video_source column in HF dataset
    VIDEO_HOME = None    # root dir for this source under VIDEO_TRACK_DATA_HOME
    DOWNLOAD_NOTE = None
    CLEANUP_FULL_VIDEOS = False  # whether to remove intermediate full_videos/ after clips are created

    @classmethod
    def _get_video_dir(cls, split):
        """Final loading directory — contains clips (or symlinked full videos)."""
        return join(cls.VIDEO_HOME, 'videos', split)

    @classmethod
    def _get_full_video_dir(cls, split):
        """Intermediate directory — full videos encoded from frames."""
        return join(cls.VIDEO_HOME, 'full_videos', split)

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        """Return path to frame directory for a given video. Override per source."""
        raise NotImplementedError

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        """Download and extract raw data so that frames are available. Override per source."""
        pass

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Build work items for full videos and clips from clip_map.

        Pipeline: frames -> full_videos/ -> videos/ (symlink or trim)

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            split: 'train' or 'test'

        Returns:
            (full_video_items, clip_items):
                full_video_items: for _create_videos (frames -> full_videos/)
                clip_items: for _create_clips (full_videos/ -> videos/)
        """
        full_video_dir = cls._get_full_video_dir(split)
        video_dir = cls._get_video_dir(split)
        full_video_items = []
        clip_items = []
        seen_videos = set()

        for clip_name, clip_info in clip_map.items():
            if isinstance(clip_info, dict):
                video_name = clip_info['video']
                fps = clip_info['fps']
            else:
                video_name = clip_name
                fps = clip_info

            frames_dir = cls._get_frames_dir(split, video_name)
            if not exists(frames_dir):
                continue

            # Stage 1: full video from frames (once per unique video)
            full_video_path = join(full_video_dir, f"{video_name}.mp4")
            if video_name not in seen_videos:
                seen_videos.add(video_name)
                if not exists(full_video_path):
                    full_video_items.append({
                        'frames_dir': frames_dir, 'output_path': full_video_path, 'fps': fps,
                    })

            # Stage 2: clip from full video -> videos/ dir
            clip_output = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_output):
                clip_item = {'source_video': full_video_path, 'output_path': clip_output, 'fps': fps}
                if clip_name == video_name:
                    clip_item['is_same_video'] = True
                else:
                    clip_item['start_frame'] = clip_info['start_frame']
                    clip_item['end_frame'] = clip_info['end_frame']
                clip_items.append(clip_item)

        return full_video_items, clip_items

    @classmethod
    def download(cls, clip_map, n_procs=1, split='train'):
        """Generic download pipeline: frames -> full_videos/ -> videos/.

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            n_procs: num workers for parallel video creation
            split: 'train' or 'test'
            cleanup_full_videos: if True, remove full_videos/ dir after clips are created
        """
        video_dir = cls._get_video_dir(split)
        full_video_dir = cls._get_full_video_dir(split)
        assert video_dir != full_video_dir, \
            f"[{cls.SOURCE_NAME}] video_dir and full_video_dir must differ: {video_dir}"

        # Step 1: Check if all clips already exist in videos/
        missing = cls._check_videos(clip_map, video_dir)
        if not missing:
            log.info(f"[{cls.SOURCE_NAME}] All {len(clip_map)} clips present for {split}, skipping.")
            return
        log.info(f"[{cls.SOURCE_NAME}] {len(missing)}/{len(clip_map)} clips missing for {split}.")

        # Step 2: Download + extract raw frames
        cls._prepare_annotation_dir(n_procs, split)

        # Step 3: Build work items
        full_video_items, clip_items = cls._build_video_work_items(clip_map, split)

        # Step 4: frames -> full_videos/
        cls._create_videos(full_video_items, n_procs)

        # Step 5: full_videos/ -> videos/ (copy or trim)
        cls._create_clips(clip_items, n_procs)

        # Step 6: Verify
        cls._check_videos(clip_map, video_dir)

        # Step 7: Optionally clean up intermediate full videos (False by default)
        if cls.CLEANUP_FULL_VIDEOS and exists(full_video_dir):
            log.info(f"[{cls.SOURCE_NAME}] Removing intermediate full_videos/ at {full_video_dir}")
            shutil.rmtree(full_video_dir)

    @classmethod
    def _verify_frames(cls, frames_dir, video_fps_map, sample_size=10):
        """Spot-check that frames dir has expected video subdirectories."""
        if not exists(frames_dir):
            return False
        sample = list(video_fps_map.keys())[:sample_size]
        for video_name in sample:
            if not exists(join(frames_dir, video_name)):
                return False
        return True

    @classmethod
    def _create_videos(cls, work_items, n_procs=1):
        """Encode frame dirs to full videos using multiprocessing."""
        if not work_items:
            log.info(f"[{cls.SOURCE_NAME}] No full videos to create.")
            return

        n = min(n_procs, len(work_items))
        log.info(f"[{cls.SOURCE_NAME}] Creating {len(work_items)} full videos with {n} workers...")
        log.info(f"[{cls.SOURCE_NAME}] Examples:")
        for item in work_items[:5]:
            log.info(f"  Frames dir: {item['frames_dir']}, Output video: {item['output_path']}, FPS: {item['fps']}")

        failed = []
        with mp.Pool(n) as pool:
            for output_path, success, error in tqdm(
                pool.imap_unordered(_encode_frames_to_video_worker, work_items),
                total=len(work_items), desc=f"[{cls.SOURCE_NAME}] Encoding full videos",
            ):
                if not success:
                    failed.append((output_path, error))
        if failed:
            log.warning(f"[{cls.SOURCE_NAME}] {len(failed)}/{len(work_items)} full videos failed, e.g.: {failed[:5]}")
        else:
            log.info(f"[{cls.SOURCE_NAME}] All {len(work_items)} full videos created successfully.")

    @classmethod
    def _create_clips(cls, work_items, n_procs=1):
        """Create clips from full videos (symlink if no trimming, ffmpeg trim otherwise)."""
        if not work_items:
            log.info(f"[{cls.SOURCE_NAME}] No clips to create.")
            return

        n = min(n_procs, len(work_items))
        log.info(f"[{cls.SOURCE_NAME}] Creating {len(work_items)} clips with {n} workers...")
        log.info(f"[{cls.SOURCE_NAME}] Examples:")
        for item in work_items[:5]:
            log.info(f"  Source video: {item['source_video']}, Output clip: {item['output_path']}, FPS: {item['fps']}, "
                     f"Start frame: {item.get('start_frame')}, End frame: {item.get('end_frame')}")

        failed = []
        with mp.Pool(n) as pool:
            for output_path, success, error in tqdm(
                pool.imap_unordered(_split_video_worker, work_items),
                total=len(work_items), desc=f"[{cls.SOURCE_NAME}] Creating clips from full videos",
            ):
                if not success:
                    failed.append((output_path, error))
        if failed:
            log.warning(f"[{cls.SOURCE_NAME}] {len(failed)}/{len(work_items)} clips failed, e.g.: {failed[:5]}")
        else:
            log.info(f"[{cls.SOURCE_NAME}] All {len(work_items)} clips created successfully.")

    @classmethod
    def _extract_videos_to_frames(cls, work_items, n_procs=1):
        """Extract videos to frames, optionally in parallel."""
        if not work_items:
            log.info("No videos to extract.")
            return

        n = min(n_procs, len(work_items))
        log.info(f"Extracting frames for {len(work_items)} videos with {n} workers...")

        failed = []
        with mp.Pool(n) as pool:
            for video_path, success, error in tqdm(
                pool.imap_unordered(_extract_frames_worker, work_items),
                total=len(work_items), desc="Extracting frames",
            ):
                if not success:
                    failed.append((video_path, error))
        if failed:
            log.warning(f"{len(failed)}/{len(work_items)} videos failed to extract, e.g.: {failed[:5]}")
        else:
            log.info(f"All {len(work_items)} videos extracted successfully.")

    @classmethod
    def _check_videos(cls, clip_map, video_dir):
        """Check that all clips in clip_map exist in video_dir."""
        missing = []
        for clip_name in clip_map:
            clip_path = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_path):
                missing.append(clip_path)
        if missing:
            log.warning(f"[{cls.SOURCE_NAME}] {len(missing)} missing clips, e.g.: {missing[:5]}")
        else:
            log.info(f"[{cls.SOURCE_NAME}] ✅ All {len(clip_map)} clips verified in {video_dir}")
        return missing

class APTv2Source(Molmo2TrackVideoSource):
    """
    APTv2: Benchmarking Animal Pose Estimation and Tracking with a Large-scale Dataset and Beyond.
    Supported splits:
        - train: for Molmo2Track
        - test: for Molmo2TrackEval
    """
    SOURCE_NAME = 'APTv2'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'APTv2')
    CLEANUP_FULL_VIDEOS = True
    MANUAL_DOWNLOAD_INSTRUCTION = """
APTv2 annotations are available on https://github.com/ViTAE-Transformer/APTv2
Download APTv2.zip from https://1drv.ms/f/s!AimBgYV7JjTlgckmLjVg2Z7B3DZXeQ?e=cHUoj9
"""

    @classmethod
    def _get_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'videos')

    @classmethod
    def _get_full_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'full_videos')

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        # APTv2 frames: APTv2/APTv2/data/{difficulty}/{category}/{vid_id}
        # video_name == clip_name == "{difficulty}_{category}_{vid_id}"
        try:
            difficulty, category, vid_id = video_name.split('_', 2)
        except ValueError:
            breakpoint()
        return join(cls.VIDEO_HOME, 'APTv2', 'data', difficulty, category, vid_id)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        annotation_dir = join(cls.VIDEO_HOME, 'APTv2', 'data')
        if exists(annotation_dir):
            log.info(f"[{cls.SOURCE_NAME}] Frames found at {annotation_dir}.")
            return
        zip_file = join(cls.VIDEO_HOME, 'APTv2.zip')
        if not file_exists(zip_file):
            log.warning(
                f"[{cls.SOURCE_NAME}] Frames not found at {annotation_dir} and APTv2.zip not found. "
                f"Please download: {cls.MANUAL_DOWNLOAD_INSTRUCTION}"
            )
            raise RuntimeError(f"Missing {annotation_dir} and {zip_file}")
        log.info(f"[{cls.SOURCE_NAME}] Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in tqdm(zf.infolist(), desc=f"[{cls.SOURCE_NAME}] Extracting"):
                zf.extract(member, cls.VIDEO_HOME)

class DanceTrackSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'dancetrack'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'DanceTrack')
    CLEANUP_FULL_VIDEOS = False

    @classmethod
    def _get_split_config(cls, split):
        """Return (zip_names, frames_root) for each split."""
        if split == 'train':
            return ['train1', 'train2'], join(cls.VIDEO_HOME, 'train')
        elif split == 'val':
            return ['val'], join(cls.VIDEO_HOME, 'val')
        elif split == 'test':
            return ['test1', 'test2'], join(cls.VIDEO_HOME, 'test')
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def _get_video_dir(cls, split):
        """Final loading directory — contains clips (or symlinked full videos)."""
        if split == 'test':
            split = 'val'
        return join(cls.VIDEO_HOME, 'videos', split)

    @classmethod
    def _get_full_video_dir(cls, split):
        """Intermediate directory — full videos encoded from frames."""
        if split == 'test':
            split = 'val'
        return join(cls.VIDEO_HOME, 'full_videos', split)

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        # DanceTrack eval uses val split frames
        data_split = 'val' if split == 'test' else split
        _, frames_root = cls._get_split_config(data_split)
        return join(frames_root, video_name, 'img1')

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        # DanceTrack eval uses val split frames
        data_split = 'val' if split == 'test' else split
        zip_names, frames_root = cls._get_split_config(data_split)

        if exists(frames_root):
            log.info(f"[{cls.SOURCE_NAME}] Frames already extracted at {frames_root}.")
            return

        # Ensure zips are present
        zip_files = [join(cls.VIDEO_HOME, f"{f}.zip") for f in zip_names]
        if not all(exists(z) for z in zip_files):
            log.info(f"[{cls.SOURCE_NAME}] Downloading dataset from HuggingFace...")
            snapshot_download(
                repo_id="noahcao/dancetrack",
                repo_type="dataset",
                local_dir=cls.VIDEO_HOME,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )

        # Extract and merge into frames_root
        for fname in zip_names:
            zip_file = join(cls.VIDEO_HOME, f"{fname}.zip")
            assert exists(zip_file), f"[{cls.SOURCE_NAME}] Expected {zip_file} not found."
            extract_dir = join(cls.VIDEO_HOME, fname)

            if not exists(extract_dir):
                log.info(f"[{cls.SOURCE_NAME}] Extracting {fname}.zip...")
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc=f"Extracting {fname}"):
                        zf.extract(member, cls.VIDEO_HOME)

            # If extracted directory is different from final frames_root, move contents and clean up
            if extract_dir != frames_root:
                os.makedirs(frames_root, exist_ok=True)
                moved = 0
                for item in os.listdir(extract_dir):
                    src = join(extract_dir, item)
                    dst = join(frames_root, item)
                    if not exists(dst):
                        shutil.move(src, dst)
                        moved += 1
                log.info(f"[{cls.SOURCE_NAME}] Merged {moved} items from {fname} into {frames_root}")
                shutil.rmtree(extract_dir, ignore_errors=True)


class SportsMOTSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'sportsmot'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'SportsMOT')
    DOWNLOAD_NOTE = 'Download from https://codalab.lisn.upsaclay.fr/competitions/12424#participate'
    CLEANUP_FULL_VIDEOS = False
    MANUAL_DOWNLOAD_INSTRUCTION = """
    1. Register for the SportsMOT challenge on CodaLab and join the competition: https://codalab.lisn.upsaclay.fr/competitions/12424#participate
    2. Download the sportsmot_publish.zip from OneDrive and place it under VIDEO_TRACK_DATA_HOME/SportsMOT:
"""

    @classmethod
    def _get_video_dir(cls, split):
        """Final loading directory — contains clips (or symlinked full videos)."""
        if split == 'test':
            split = 'val'
        return join(cls.VIDEO_HOME, 'videos', split)

    @classmethod
    def _get_full_video_dir(cls, split):
        """Intermediate directory — full videos encoded from frames."""
        if split == 'test':
            split = 'val'
        return join(cls.VIDEO_HOME, 'full_videos', split)

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        # SportsMOT eval uses val split frames
        data_split = 'val' if split == 'test' else split
        return join(cls.VIDEO_HOME, 'sportsmot_publish', 'dataset', data_split, video_name, 'img1')

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        # Single zip contains all splits — check if already extracted
        dataset_dir = join(cls.VIDEO_HOME, 'sportsmot_publish', 'dataset')
        if exists(dataset_dir):
            log.info(f"[{cls.SOURCE_NAME}] Frames already extracted at {dataset_dir}.")
            return
        zip_file = join(cls.VIDEO_HOME, 'sportsmot_publish.zip')
        if not exists(zip_file):
            log.warning(f"[{cls.SOURCE_NAME}] {zip_file} not found. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
            raise RuntimeError(f"Missing {zip_file}")
        log.info(f"[{cls.SOURCE_NAME}] Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in tqdm(zf.infolist(), desc=f"[{cls.SOURCE_NAME}] Extracting"):
                zf.extract(member, cls.VIDEO_HOME)

class PersonPath22Source(Molmo2TrackVideoSource):
    SOURCE_NAME = 'personpath22'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'personpath22')
    CLEANUP_FULL_VIDEOS = False
    MANUAL_DOWNLOAD_INSTRUCTION = """
    1. Data download instruction available at: https://github.com/amazon-science/tracking-dataset
    2. Run their download script to download and extract videos: https://github.com/amazon-science/tracking-dataset/blob/main/download.py
    3. Place the extracted videos at VIDEO_TRACK_DATA_HOME/personpath22/full_videos
"""

    @classmethod
    def _get_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'videos', split)

    @classmethod
    def _get_full_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'full_videos')

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Build work items for full videos and clips from clip_map.

        Pipeline: frames -> full_videos/ -> videos/ (symlink or trim)

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            split: 'train' or 'test'

        Returns:
            (full_video_items, clip_items):
                full_video_items: for _create_videos (frames -> full_videos/)
                clip_items: for _create_clips (full_videos/ -> videos/)
        """
        full_video_dir = cls._get_full_video_dir(split)
        video_dir = cls._get_video_dir(split)
        full_video_items = []
        clip_items = []

        for clip_name, clip_info in clip_map.items():
            if isinstance(clip_info, dict):
                video_name = clip_info['video']
                fps = clip_info['fps']
            else:
                video_name = clip_name
                fps = clip_info

            # Get full video from download scripts
            full_video_path = join(full_video_dir, f"{video_name}.mp4")

            # Make clip from full video -> videos/ dir
            clip_output = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_output):
                clip_item = {'source_video': full_video_path, 'output_path': clip_output, 'fps': fps}
                if clip_name == video_name:
                    clip_item['is_same_video'] = True
                else:
                    clip_item['start_frame'] = clip_info['start_frame']
                    clip_item['end_frame'] = clip_info['end_frame']
                clip_items.append(clip_item)

        return full_video_items, clip_items


class SAVSource(Molmo2TrackVideoSource):
    """Segment Anything Video. Train: extract mp4->frames->reencode at 6fps. Test: extract tar->frames->encode."""
    SOURCE_NAME = 'sav'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'sav')
    ORIGINAL_FPS = 24  # we subsample to 6 fps
    VIDEO_FPS = 6
    CLEANUP_FULL_VIDEOS = True

    MANUAL_DOWNLOAD_INSTRUCTION = """
1. Download frames from https://ai.meta.com/datasets/segment-anything-video/
2. Place download following tar files to VIDEO_TRACK_DATA_HOME/sav:
    Train: sav_{000..003}.tar
    Eval: sav_test.tar
3. The download script will extract frames and recreate videos at 6 fps.
"""

    @classmethod
    def _get_video_dir(cls, split):
        return join(cls.VIDEO_HOME, f"sav_{split}", f'videos_fps{cls.VIDEO_FPS}')

    @classmethod
    def _get_full_video_dir(cls, split):
        return join(cls.VIDEO_HOME, f"sav_{split}", f'full_videos_fps{cls.VIDEO_FPS}')

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, f"sav_{split}", f'JPEGImages_{cls.ORIGINAL_FPS}fps', video_name)

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Override to add native_fps for frame subsampling on full video encoding."""
        full_video_items, clip_items = super()._build_video_work_items(clip_map, split)
        for item in full_video_items:
            item['native_fps'] = cls.ORIGINAL_FPS
        # clip_items trim from the already-subsampled full video, no native_fps needed
        return full_video_items, clip_items

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        """Extract tar files and decode mp4s to frames."""
        if split == 'train':
            cls._prepare_train_frames(n_procs)
        else:
            cls._prepare_test_frames(split, n_procs)

    @classmethod
    def _prepare_train_frames(cls, n_procs=1):
        """Extract sav_000..003.tar -> decode mp4s -> JPEGImages_24fps/"""
        train_splits = [f"sav_{i:03d}" for i in range(4)]
        for train_split in train_splits:
            video_dir_path = join(cls.VIDEO_HOME, 'sav_train', train_split)
            if not exists(video_dir_path):
                tar_file = join(cls.VIDEO_HOME, f"{train_split}.tar")
                if not exists(tar_file):
                    raise RuntimeError(
                        f"Missing {video_dir_path} and {tar_file}. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
                log.info(f"Extracting {tar_file}...")
                subprocess.run(["tar", "--warning=no-unknown-keyword", "-xf", tar_file, "-C", cls.VIDEO_HOME,
                                "--checkpoint=10000", "--checkpoint-action=echo=%c records"], check=True)

            # Decode mp4s to frames
            extract_items = []
            for video_file in tqdm(os.listdir(video_dir_path), desc=f"Scanning {train_split}", leave=False):
                if video_file.endswith(".mp4"):
                    video_name = os.path.splitext(video_file)[0]
                    frames_dir = cls._get_frames_dir('train', video_name)
                    if not exists(frames_dir):
                        extract_items.append({'video_path': join(video_dir_path, video_file), 'out_folder': frames_dir})
            cls._extract_videos_to_frames(extract_items, n_procs)

    @classmethod
    def _prepare_test_frames(cls, split, n_procs=1):
        """Extract sav_{split}.tar to get JPEGImages_24fps/"""
        local_frames_dir = join(cls.VIDEO_HOME, f"sav_{split}", f"JPEGImages_{cls.ORIGINAL_FPS}fps")
        if exists(local_frames_dir):
            return
        tar_file = join(cls.VIDEO_HOME, f"sav_{split}.tar")
        if not exists(tar_file):
            raise RuntimeError(f"Missing {local_frames_dir} and {tar_file}. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
        log.info(f"Extracting {tar_file}...")
        subprocess.run(["tar", "--warning=no-unknown-keyword", "-xf", tar_file, "-C", cls.VIDEO_HOME,
                        "--checkpoint=10000", "--checkpoint-action=echo=%c records"], check=True)

class MOSESource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'mose'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'MOSE')
    CLEANUP_FULL_VIDEOS = True

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'MOSE_release', 'train', 'JPEGImages', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        frames_root = join(cls.VIDEO_HOME, 'MOSE_release', 'train', 'JPEGImages')
        if exists(frames_root):
            log.info(f"[{cls.SOURCE_NAME}] Frames already extracted at {frames_root}")
            return

        # Download from HuggingFace
        log.info(f"[{cls.SOURCE_NAME}] Downloading from HuggingFace...")
        snapshot_download(
            repo_id="FudanCVL/MOSE",
            repo_type="dataset",
            local_dir=cls.VIDEO_HOME,
            local_dir_use_symlinks=False,
            max_workers=n_procs,
        )

        # Unzip MOSE_release.zip
        release_dir = join(cls.VIDEO_HOME, 'MOSE_release')
        if not exists(release_dir):
            zip_path = join(cls.VIDEO_HOME, 'MOSE_release.zip')
            log.info(f"[{cls.SOURCE_NAME}] Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc="MOSE_release.zip", leave=False):
                    zf.extract(member, cls.VIDEO_HOME)

        # Untar train.tar.gz inside MOSE_release/
        train_dir = join(release_dir, 'train')
        if not exists(train_dir):
            tar_path = join(release_dir, 'train.tar.gz')
            if not exists(tar_path):
                raise RuntimeError(f"[{cls.SOURCE_NAME}] Missing {tar_path} after unzipping")
            log.info(f"[{cls.SOURCE_NAME}] Extracting {tar_path}...")
            subprocess.run(
                ["tar", "--warning=no-unknown-keyword", "-xzf", tar_path, "-C", release_dir,
                 "--checkpoint=1000", "--checkpoint-action=echo=%c records"],
                check=True,
            )


class MOSEv2Source(Molmo2TrackVideoSource):
    SOURCE_NAME = 'mosev2'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'MOSEv2')
    CLEANUP_FULL_VIDEOS = True

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'train', 'JPEGImages', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        frames_root = join(cls.VIDEO_HOME, 'train', 'JPEGImages')
        if exists(frames_root):
            log.info(f"[{cls.SOURCE_NAME}] Frames already extracted at {frames_root}")
            return

        # Check if split tar parts already exist (from prior download)
        tar_path = join(cls.VIDEO_HOME, 'train.tar.gz')
        tar_parts = sorted(glob(join(cls.VIDEO_HOME, 'train.tar.gz.*')))

        if not tar_parts:
            # Download from HuggingFace — repo has train.tar.gz.aa, .ab, .ac, etc.
            log.info(f"[{cls.SOURCE_NAME}] Downloading from HuggingFace (FudanCVL/MOSEv2)...")
            snapshot_download(
                repo_id="FudanCVL/MOSEv2",
                repo_type="dataset",
                local_dir=cls.VIDEO_HOME,
                local_dir_use_symlinks=False,
                max_workers=n_procs,
            )
            tar_parts = sorted(glob(join(cls.VIDEO_HOME, 'train.tar.gz.*')))

        # Concatenate split tar parts into single tar
        if tar_parts and not exists(tar_path):
            log.info(f"[{cls.SOURCE_NAME}] Concatenating {len(tar_parts)} split tar parts to create {tar_path}...")
            subprocess.run(
                ["cat"] + tar_parts,
                stdout=open(tar_path, 'wb'),
                check=True,
                )

        # Extract train.tar.gz → train/JPEGImages/
        if exists(tar_path):
            log.info(f"[{cls.SOURCE_NAME}] Extracting {tar_path}...")
            subprocess.run(
                ["tar", "--warning=no-unknown-keyword", "-xzf", tar_path, "-C", cls.VIDEO_HOME,
                 "--checkpoint=1000", "--checkpoint-action=echo=%c records"],
                check=True,
            )

        if not exists(frames_root):
            raise RuntimeError(f"[{cls.SOURCE_NAME}] Frames not found at {frames_root} after download and extraction")


class VIPSegSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'vipseg'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'VIPSeg')
    CLEANUP_FULL_VIDEOS = True
    VIDEO_URL = "https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ/view?usp=sharing"
    MANUAL_DOWNLOAD_INSTRUCTION = """
    Download from github repo: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
    Follow instruction to change VIPseg to 720P and coco format: 
    https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/?tab=readme-ov-file#change-vipseg-to-720p-and-coco-format
    
    1. Download VIPSeg.tar from google drive:  https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ/view?usp=sharing
    2. tar -xf VIPSeg.tar -C VIDEO_TRACK_DATA_HOME/VIPSeg
    3. Change to 720P by running https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/change2_720p.py
    Place them under VIDEO_TRACK_DATA_HOME/VIPSeg_720P/
"""
    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'VIPSeg_720P', 'images', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        frames_root = join(cls.VIDEO_HOME, 'VIPSeg_720P', 'images')
        if not exists(frames_root):
            log.info(f"[{cls.SOURCE_NAME}] Frames not found at {frames_root}. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
            raise RuntimeError(f"Missing {frames_root}. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
        else:
            log.info(f"[{cls.SOURCE_NAME}] Frames found at {frames_root}.")


class AnimalTrackSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'animaltrack'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'animaltrack')
    VIDEO_URL = "https://drive.google.com/file/d/1SKijkh5ddCdIoVhpEnVvtflvOWvwo1EN/view?usp=drive_link"
    CLEANUP_FULL_VIDEOS = True

    @classmethod
    def _get_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'videos_all')

    @classmethod
    def download(cls, clip_map, n_procs=1, split='train'):
        """Generic download pipeline: frames -> full_videos/ -> videos/.

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            n_procs: num workers for parallel video creation
            split: 'train' or 'test'
            cleanup_full_videos: if True, remove full_videos/ dir after clips are created
        """
        video_dir = cls._get_video_dir(split)

        # Step 1: Check if all clips already exist in videos/
        missing = cls._check_videos(clip_map, video_dir)
        if not missing:
            log.info(f"[{cls.SOURCE_NAME}] All {len(clip_map)} clips present for {split}, skipping.")
            return
        log.info(f"[{cls.SOURCE_NAME}] {len(missing)}/{len(clip_map)} clips missing for {split}.")

        # Step 2: Download videos (animaltrack already provides videos, no frame-to-video conversion needed)
        cls._prepare_annotation_dir(n_procs, split)

        # Step 3: Verify
        cls._check_videos(clip_map, video_dir)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):

        maybe_download_and_unzip(
            cls.VIDEO_HOME,
            cls.VIDEO_URL,
            expected_dir="videos_all",
        )


class BFTSource(Molmo2TrackVideoSource):
    """
    NetTrack: Tracking Highly Dynamic Objects with a Net
    Source: https://george-zhuang.github.io/nettrack/
    Consist of Bird Flock Tracking (BFT) videos.
    """
    SOURCE_NAME = 'bft'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'bft')
    DOWNLOAD_NOTE = 'Download from https://george-zhuang.github.io/nettrack/'
    CLEANUP_FULL_VIDEOS = True
    VIDEO_URL = "https://drive.google.com/file/d/1iEebl-2yPjapQByOotoLG_0ud_1q_hZs/view?usp=drive_link"
    MANUAL_DOWNLOAD_INSTRUCTION = """
1. Download train.zip from https://george-zhuang.github.io/nettrack/
2. Place the downloaded videos at VIDEO_TRACK_DATA_HOME/bft
"""

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'JPEGImages', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        log.info(f"Downloading BFT frames to {cls.VIDEO_HOME}...")
        maybe_download_and_unzip(
            join(cls.VIDEO_HOME, 'JPEGImages'),
            cls.VIDEO_URL,
            expected_dir="JPEGImages",
        )

class SoccerNetSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'soccernet'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'SoccerNet')
    CLEANUP_FULL_VIDEOS = False
    MANUAL_DOWNLOAD_INSTRUCTION = """
1. Register and sign NDA at https://www.soccer-net.org/data
2. Download their tracking videos using their pip tool: https://pypi.org/project/SoccerNet/
    >>> from SoccerNet.Downloader import SoccerNetDownloader
    >>> mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="VIDEO_TRACK_DATA_HOME/SoccerNet")
    >>> mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train"])
"""

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'tracking', split, video_name, 'img1')

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        annotation_dir = join(cls.VIDEO_HOME, 'tracking')
        if not exists(annotation_dir):
            log.warning(f"[{cls.SOURCE_NAME}] Annotation preparation requires manual download due to NDA. {cls.DOWNLOAD_NOTE}")
            raise NotImplementedError(f"Manual download required for {cls.SOURCE_NAME}. {cls.DOWNLOAD_NOTE}")

        # Unzip tracking.zip
        if not exists(join(annotation_dir, split)):
            split_dir = join(annotation_dir, split)
            log.info(f"[{cls.SOURCE_NAME}] {split_dir} directory not found for {split}...")
            zip_file = join(annotation_dir, f"{split}.zip")
            log.info(f"[{cls.SOURCE_NAME}] Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                for member in tqdm(zf.infolist(), desc=f"[{cls.SOURCE_NAME}] Extracting {split}.zip", leave=False):
                    zf.extract(member, annotation_dir)
        else:
            log.info(f"[{cls.SOURCE_NAME}] Frames for split: {split} already extracted at {annotation_dir}.")


class TeamTrackSource(Molmo2TrackVideoSource):
    SOURCE_NAME = 'teamtrack'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'TeamTrack')
    CLEANUP_FULL_VIDEOS = True
    VIDEO_URL = "https://drive.google.com/file/d/12KOMgl9BFirmPVHbbUTUWacc9vLG94er/view?usp=drive_link"

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        """Download and extract teamtrack.zip if teamtrack directory doesn't already exist"""
        maybe_download_and_unzip(
            cls.VIDEO_HOME,
            cls.VIDEO_URL,
            expected_dir="teamtrack",
        )

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Build work items for full videos and clips from clip_map.

        Pipeline: frames -> full_videos/ -> videos/ (symlink or trim)

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            split: 'train' or 'test'

        Returns:
            (full_video_items, clip_items):
                full_video_items: for _create_videos (frames -> full_videos/)
                clip_items: for _create_clips (full_videos/ -> videos/)
        """
        video_dir = cls._get_video_dir(split)
        full_video_items = []
        clip_items = []

        # teamtrack-mot/soccer_side/train/F_20200220_1_0180_0210/img1.mp4
        all_videos = glob(join(cls.VIDEO_HOME, f'teamtrack/*/{split}/videos/*.mp4'))
        video_name_map = {os.path.splitext(os.path.basename(v))[0]: v for v in all_videos}

        for clip_name, clip_info in clip_map.items():
            if isinstance(clip_info, dict):
                video_name = clip_info['video']
                fps = clip_info['fps']
            else:
                video_name = clip_name
                fps = clip_info

            # Get full video path from downloaded data
            full_video_path = video_name_map[video_name]

            # Make clip from full video -> videos/ dir
            clip_output = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_output):
                clip_item = {'source_video': full_video_path, 'output_path': clip_output, 'fps': fps}
                if clip_name == video_name:
                    clip_item['is_same_video'] = True
                else:
                    clip_item['start_frame'] = clip_info['start_frame']
                    clip_item['end_frame'] = clip_info['end_frame']
                clip_items.append(clip_item)

        return full_video_items, clip_items


class MOT20Source(Molmo2TrackVideoSource):
    """
    MOT20: Multiple Object Tracking 2020. https://motchallenge.net/data/MOT20/
    We use all MOT15/MOT16/MOT17/MOT20 training sets
    """
    SOURCE_NAME = 'mot2020'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'MOT20')
    CLEANUP_FULL_VIDEOS = False

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        if 'MOT20' in video_name:
            data_dir = 'MOT20'
        elif 'MOT17' in video_name:
            data_dir = 'MOT17'
        elif 'MOT16' in video_name:
            data_dir = 'MOT16'
        else:
            data_dir = 'MOT15'
        return join(cls.VIDEO_HOME, data_dir, split, video_name, 'img1')

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        for yr in ['15', '16', '17', '20']:
            video_url = f"https://motchallenge.net/data/MOT{yr}.zip"
            log.info(f"Downloading MOT{yr} frames to {cls.VIDEO_HOME}...")
            out_dir = cls.VIDEO_HOME
            if yr == '16':
                out_dir = join(cls.VIDEO_HOME, 'MOT16')  # MOT16 has train/test split inside
                if exists(out_dir):
                    log.info(f"MOT16 frames already extracted at {out_dir}.")
                    continue
                maybe_download_and_unzip(
                    out_dir,
                    video_url,
                )
            else:
                maybe_download_and_unzip(
                    out_dir,
                    video_url,
                    expected_dir=f"MOT{yr}",
                )


class BDD100KSource(Molmo2TrackVideoSource):
    """
    BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.
    We use train_00 split as our training data.
    """
    SOURCE_NAME = 'bdd100k'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'bdd100k')
    VIDEO_URL = 'http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_00.zip'
    CLEANUP_FULL_VIDEOS = False

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        log.info(f"Downloading BDD100K frames to {cls.VIDEO_HOME}...")
        maybe_download_and_unzip(
            cls.VIDEO_HOME,
            cls.VIDEO_URL,
            expected_dir="bdd100k",
        )

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Build work items for full videos and clips from clip_map.

        Pipeline: frames -> full_videos/ -> videos/ (symlink or trim)

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            split: 'train' or 'test'

        Returns:
            (full_video_items, clip_items):
                full_video_items: for _create_videos (frames -> full_videos/)
                clip_items: for _create_clips (full_videos/ -> videos/)
        """
        video_dir = cls._get_video_dir(split)
        full_video_items = []
        clip_items = []

        for clip_name, clip_info in clip_map.items():
            if isinstance(clip_info, dict):
                video_name = clip_info['video']
                fps = clip_info['fps']
            else:
                video_name = clip_name
                fps = clip_info

            # Get full video path from downloaded data
            full_video_path = join(cls.VIDEO_HOME, f'bdd100k/videos/{split}/{video_name}.mov')

            # Make clip from full video -> videos/ dir
            clip_output = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_output):
                clip_item = {'source_video': full_video_path, 'output_path': clip_output, 'fps': fps}
                if clip_name == video_name:
                    clip_item['is_same_video'] = True
                else:
                    clip_item['start_frame'] = clip_info['start_frame']
                    clip_item['end_frame'] = clip_info['end_frame']
                clip_items.append(clip_item)

        return full_video_items, clip_items


class UAVDTSource(Molmo2TrackVideoSource):
    """
    UAVDT: Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking.
    https://sites.google.com/view/grli-uavdt/
    
    Download UAVDT-Benchmark-M Dataset from the above link.
    """
    SOURCE_NAME = 'uavdt'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'UAVDT')
    VIDEO_URL = "https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view"
    CLEANUP_FULL_VIDEOS = False

    @classmethod
    def _get_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'videos')

    @classmethod
    def _get_full_video_dir(cls, split):
        return join(cls.VIDEO_HOME, 'full_videos')

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        return join(cls.VIDEO_HOME, 'UAV-benchmark-M', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        log.info(f"Downloading UAVDT frames to {cls.VIDEO_HOME}...")
        maybe_download_and_unzip(
            cls.VIDEO_HOME,
            cls.VIDEO_URL,
            expected_dir="UAV-benchmark-M",
        )


class SeaDronesSeeSource(Molmo2TrackVideoSource):
    """
    SeaDronesSee: A Dataset for Object Detection and Tracking in Maritime Environments.
    https://seadronessee.cs.uni-tuebingen.de/dataset

    Raw data is a flat directory of .jpg images. The annotation JSON groups images
    into videos by video_id. We parse the annotation to create per-video frame
    directories with symlinks, then encode to mp4.

    Some long videos are split into sub-videos in the HF data (e.g. DJI_0069-0, DJI_0069-1).
    _get_frames_dir strips the '-N' suffix so sub-videos share the same frame directory;
    the base pipeline creates a full video from all frames, then trims clips via start_frame/end_frame.

    Duplicate video names (e.g. two DJI_0001 from different drones) are disambiguated
    with a '_2' suffix matching the HF naming convention.
    """
    SOURCE_NAME = 'seadrones'
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'SeaDronesSee')
    VIDEO_URL = 'https://cloud.cs.uni-tuebingen.de/public.php/dav/files/W4ztazMxqfHdYWA/SeaDronesSee_MOT_jpg_compressed.zip'
    ANNOTATION_URL = "https://cloud.cs.uni-tuebingen.de/public.php/dav/files/W4ztazMxqfHdYWA/annotations/instances_train_objects_in_water.json"
    DOWNLOAD_NOTE = "Download 'Multi-Object Tracking' from https://seadronessee.cs.uni-tuebingen.de/dataset"
    CLEANUP_FULL_VIDEOS = False

    @classmethod
    def _get_frames_dir(cls, split, video_name):
        # Strip sub-video suffix: DJI_0069-0 -> DJI_0069, DJI_0001_2-3 -> DJI_0001_2
        return join(cls.VIDEO_HOME, 'frames', video_name)

    @classmethod
    def _prepare_annotation_dir(cls, n_procs=1, split='train'):
        frames_root = join(cls.VIDEO_HOME, 'frames')
        if exists(frames_root):
            log.info(f"[{cls.SOURCE_NAME}] Frame directories already exist at {frames_root}")
            return

        # Ensure raw data is downloaded
        img_dir = join(cls.VIDEO_HOME, 'Compressed', 'train')
        if not exists(img_dir):
            log.info(f"[{cls.SOURCE_NAME}] Downloading compressed images...")
            maybe_download_and_unzip(
                cls.VIDEO_HOME,
                cls.VIDEO_URL,
                expected_dir="Compressed",
            )

        ann_path = join(cls.VIDEO_HOME, 'instances_train_objects_in_water.json')
        if not exists(ann_path):
            maybe_download_file(cls.ANNOTATION_URL, ann_path)

        # Parse annotation: group flat images into per-video frame directories
        log.info(f"[{cls.SOURCE_NAME}] Parsing annotation and creating per-video frame directories...")
        with open(ann_path) as f:
            ann = json.load(f)

        # Build video_id -> unique name mapping (disambiguate duplicate base names with _2 suffix)
        vid_names = {}
        name_counts = {}
        for v in sorted(ann['videos'], key=lambda x: x['id']):
            # Extract base name: /data/input/.../DJI_0057.MP4 -> DJI_0057
            raw_name = v.get('name:', v.get('name', ''))
            base_name = raw_name.split('/')[-1].rsplit('.', 1)[0]
            if base_name in name_counts:
                name_counts[base_name] += 1
                unique_name = f"{base_name}_{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                unique_name = base_name
            vid_names[v['id']] = unique_name

        # Group images by video_id
        from collections import defaultdict
        vid_frames = defaultdict(list)
        for img in ann['images']:
            vid_frames[img['video_id']].append(img)

        # Create symlinked frame directories
        os.makedirs(frames_root, exist_ok=True)
        for vid_id, frames in tqdm(vid_frames.items(), desc="Creating frame directories"):
            if vid_id not in vid_names:
                log.warning(f"[{cls.SOURCE_NAME}] video_id {vid_id} has {len(frames)} images but no entry in videos list, skipping")
                continue
            vid_name = vid_names[vid_id]
            vid_dir = join(frames_root, vid_name)
            if exists(vid_dir):
                continue
            os.makedirs(vid_dir, exist_ok=True)

            frames.sort(key=lambda x: x['frame_index'])
            for i, frame in enumerate(frames):
                # Annotation says .png but actual files are .jpg
                src_name = frame['file_name'].replace('.png', '.jpg')
                src = join(img_dir, src_name)
                dst = join(vid_dir, f"{i:05d}.jpg")
                if not exists(dst) and exists(src):
                    os.symlink(os.path.abspath(src), dst)

        log.info(f"[{cls.SOURCE_NAME}] Created {len(vid_frames)} video frame directories in {frames_root}")
        log.info(f"[{cls.SOURCE_NAME}] Video name mapping: {vid_names}")

    @classmethod
    def _build_video_work_items(cls, clip_map, split):
        """Build work items for full videos and clips from clip_map.

        Pipeline: frames -> full_videos/ -> videos/ (symlink or trim)

        Args:
            clip_map: {clip_name: {video, fps, start_frame, end_frame}} or {video_name: fps}
            split: 'train' or 'test'

        Returns:
            (full_video_items, clip_items):
                full_video_items: for _create_videos (frames -> full_videos/)
                clip_items: for _create_clips (full_videos/ -> videos/)
        """
        full_video_dir = cls._get_full_video_dir(split)
        video_dir = cls._get_video_dir(split)
        full_video_items = []
        clip_items = []
        seen_videos = set()

        for clip_name, clip_info in clip_map.items():
            if isinstance(clip_info, dict):
                video_name = clip_info['video']
                fps = clip_info['fps']
            else:
                video_name = clip_name
                fps = clip_info

            video_name = video_name.rsplit('-', 1)[0]
            frames_dir = cls._get_frames_dir(split, video_name)
            if not exists(frames_dir):
                continue

            # Stage 1: full video from frames (once per unique video)
            full_video_path = join(full_video_dir, f"{video_name}.mp4")
            if video_name not in seen_videos:
                seen_videos.add(video_name)
                if not exists(full_video_path):
                    full_video_items.append({
                        'frames_dir': frames_dir, 'output_path': full_video_path, 'fps': fps,
                    })

            # Stage 2: clip from full video -> videos/ dir
            clip_output = join(video_dir, f"{clip_name}.mp4")
            if not exists(clip_output):
                clip_item = {'source_video': full_video_path, 'output_path': clip_output, 'fps': fps}
                if clip_name == video_name:
                    clip_item['is_same_video'] = True
                else:
                    clip_item['start_frame'] = clip_info['start_frame']
                    clip_item['end_frame'] = clip_info['end_frame']
                clip_items.append(clip_item)

        return full_video_items, clip_items


class MolmoPointTrackAny(TrackingDataset):
    """Point tracking dataset over natural videos from YouTube and MammalNet.

    Loads annotations from allenai/MolmoPoint-TrackAny on HuggingFace. Each example
    contains a natural-language expression, per-frame point trajectories, and video
    metadata. Two sampling-FPS variants (1 and 2) are concatenated into a single
    train split.

    Video sources (column ``video_source``):
        - ``"youtube"``: hosted on GCS (requester-pays), use the URL mapping JSON
          ``molmo_point_track_youtube_id_to_urls_mapping.json`` to download.
        - ``"MammalNet"``: auto-downloaded from mammalnet.s3.amazonaws.com.

    Expected directory layout under VIDEO_HOME::

        {VIDEO_HOME}/youtube-cc/{video_name}.{ext}   (ext may be .mp4, .webm, .mkv, etc.)
        {VIDEO_HOME}/MammalNet/trimmed_video/{video_name}.mp4

    See also: scripts/molmopoint_trackany_readme_draft.md (HuggingFace dataset card)
    """
    HF_SOURCE = "allenai/MolmoPoint-TrackAny"
    DATASET_NAME = "molmopoint-trackany"
    VIDEO_HOME = join(VIDEO_DATA_HOME) # shares same video as Molmo2VideoPoint, hence VIDEO_DATA_HOME instead of VIDEO_TRACK_DATA_HOME
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
    }
    MANUAL_DOWNLOAD_INSTRUCTION = """
This dataset has two video sources:

1. YouTube videos (video_source="youtube"):
   Follow the Molmo2-VideoPoint download pattern using GCS requester-pays.
   Details available in: https://huggingface.co/datasets/allenai/MolmoPoint-TrackAny#video-download
   a) Get the URL mapping: molmo_point_track_youtube_id_to_urls_mapping.json
   b) Set up a GCS project with billing: https://cloud.google.com/storage/docs/requester-pays
   c) Download videos and place them at: {VIDEO_DATA_HOME}/youtube-cc/{video_path}
   Alternatively, bulk-copy from our bucket:
      gsutil -u YOUR_PROJECT cp -r gs://molmo2_videos/track_synthetic/* gs://your-bucket/

2. MammalNet videos (video_source="MammalNet"):
   Auto-downloaded during MolmoPointTrackAny.download() from:
      https://mammalnet.s3.amazonaws.com/trimmed_video.tar.gz
   Extracted to: {VIDEO_DATA_HOME}/MammalNet/
"""

    @classmethod
    def _resolve_video_path(cls, video_id, video_path, video_source):
        """Look up the video path from the pre-built cache.

        Args:
            video_id: the `video` column value from HF (video_id, no extension)
            video_source: "youtube" or "MammalNet"

        Returns the absolute path if found, None otherwise.
        """
        if video_source in ("MammalNet", "mammalnet"):
            return join(cls.VIDEO_HOME, "MammalNet", "trimmed_video", f"{video_id}.mp4")

        # YouTube: look up from URL mapping cache
        return join(cls.VIDEO_HOME, "youtube-cc", video_path)

    @classmethod
    def _load_all_dataset_and_fps(cls, overwrite_cache=True):
        """Load/cache annotations across all tasks/splits, return {(data_split, video_name): fps}.

        video_name is the video identifier without file extension.
        overwrite_cache: re-download from HF even if local cache exists (pass True from download()).
        """
        video_fps = {}
        seen = set()
        for task in cls.TASKS:
            for data_split in set(cls.SPLIT_MAP.values()):
                if (task, data_split) in seen:
                    continue
                seen.add((task, data_split))
                try:
                    local_dir = cls._get_local_dir(task, data_split)
                    config = cls._get_hf_config(task)
                    ds = _load_hf_dataset(cls.HF_SOURCE, data_split, local_name=local_dir,
                                          config=config, overwrite_cache=overwrite_cache)
                    ds = ds.select_columns(["video", "video_path", "fps", "video_source"])
                except Exception as e:
                    log.warning(f"Could not load {cls.HF_SOURCE}/{config}/{data_split}: {e}")
                    continue
                for ex in ds:
                    key = (data_split, ex['video'], ex['video_path'], ex['video_source'])
                    fps = ex['fps']
                    if key in video_fps:
                        assert video_fps[key] == fps, f"Conflicting FPS for {key}: {video_fps[key]} vs {fps}"
                    video_fps[key] = fps
        return video_fps


    @classmethod
    def download(cls, n_procs=1):
        """Generic download pipeline. Override _prepare_annotation_dir and
        _get_frames_dir to customize; override download() entirely for
        non-standard flows (e.g. pre-existing videos, chunked downloads)."""
        # Step 1: What videos do we need?
        video_fps_map = cls._load_all_dataset_and_fps()

        # Step 2: Which are missing?
        missing = cls._check_videos(video_fps_map)
        if missing:
            log.info(f"[{cls.DATASET_NAME}] {len(missing)}/{len(video_fps_map)} videos missing.")

            # Step 3: Download videos from source(s)
            yt_dir = join(cls.VIDEO_HOME, 'youtube-cc')
            if not exists(yt_dir):
                log.info(f"Not found {yt_dir}. Assuming youtube videos not downloaded yet")
                log.info(f"Please follow manual download instruction to download videos to {cls.VIDEO_HOME}. {cls.MANUAL_DOWNLOAD_INSTRUCTION}")
            else:
                log.info(f"Found existing YouTube videos at {yt_dir}, skipping YouTube download.")

            mammal_dir = join(cls.VIDEO_HOME, 'MammalNet', )
            if not exists(join(mammal_dir, "trimmed_video")):
                log.info(f"Downloading MammalNet trimmed videos to {mammal_dir}...")

                # note: original repo has typo as "trimmed_videos.tar.gz" but actual file is "trimmed_video.tar.gz"

                tar_path = join(mammal_dir, "trimmed_video.tar.gz")
                if not exists(tar_path):
                    maybe_download_file(
                        "https://mammalnet.s3.amazonaws.com/trimmed_video.tar.gz",
                        tar_path,
                    )
                else:
                    log.info(f"Found existing tar file at {tar_path}, skipping download.")

                log.info(f"Extracting MammalNet trimmed videos...")
                subprocess.run(["tar", "-xzf", tar_path, "-C", mammal_dir,
                                "--checkpoint=10000", "--checkpoint-action=echo=#%u files extracted"], check=True)

            cls._check_videos(video_fps_map)

    @classmethod
    def _check_videos(cls, video_fps):
        """Check that all videos referenced in annotations exist on disk.

        Uses _resolve_video_path to handle varying extensions for YouTube videos.
        Builds the path cache from the URL mapping JSON if not already built.
        """
        missing = []
        for (data_split, video_id, video_path, video_source), fps in video_fps.items():
            path = cls._resolve_video_path(video_id, video_path, video_source)
            if path is None or not exists(path):
                missing.append(video_path)
        if missing:
            log.warning(f"[{cls.DATASET_NAME}] {len(missing)} missing videos, e.g.: {missing[:20]}")
        else:
            log.info(f"[{cls.DATASET_NAME}] ✅ All {len(video_fps)} unique instances verified")
        return missing

    def get(self, idx, rng):
        ex = self.data[idx]
        video_fps = ex["fps"]
        video_source = ex["video_source"]

        video_path = self._resolve_video_path(ex['video'], ex['video_path'], video_source)
        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['expression'],
            'w': ex['width'],
            'h': ex['height'],
            'video_fps': video_fps,
            'video': ex['video'],
            'video_source': video_source,
        }

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': self.sampling_fps or ex['sampling_fps'],
            }

        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': ex['sampling_fps'],
            'metadata': metadata,
        }


class MolmoPointTrackSyn(TrackingDataset):
    """Synthetic point tracking dataset.

    Loads annotations from allenai/MolmoPoint-TrackSyn on HuggingFace.
    Videos are stored as synthetic_tracks.tar in the same HF repo.

    After extraction, videos live at::

        {VIDEO_HOME}/static-camera/{run_dir}/{video_file}.mp4

    The HF ``video`` column stores the relative path (no extension), so
    the base class ``get()`` resolves via ``ex['video'] + '.mp4'``.
    """
    HF_SOURCE = "allenai/MolmoPoint-TrackSyn"
    DATASET_NAME = "molmopoint-tracksyn"
    VIDEO_HOME = join(VIDEO_TRACK_DATA_HOME, 'MolmoPoint-TrackSyn')
    VIDEO_TAR = "synthetic_tracks.tar"
    TASKS = ["track"]
    SPLIT_MAP = {
        "train": "train",
    }

    @classmethod
    def _get_video_dir(cls, data_split):
        """Videos are directly under VIDEO_HOME (no per-split subdirs)."""
        return cls.VIDEO_HOME

    @classmethod
    def download(cls, n_procs=1, sources=None, configs=None):
        """Download annotations and synthetic videos from HuggingFace."""
        video_fps_map = cls._load_all_dataset_and_fps()

        missing = cls._check_videos(video_fps_map)
        if missing:
            log.info(f"[{cls.DATASET_NAME}] {len(missing)}/{len(video_fps_map)} videos missing.")

            tar_path = join(cls.VIDEO_HOME, cls.VIDEO_TAR)
            if not exists(tar_path):
                log.info(f"[{cls.DATASET_NAME}] Downloading {cls.VIDEO_TAR} from {cls.HF_SOURCE}...")
                snapshot_download(
                    repo_id=cls.HF_SOURCE,
                    allow_patterns=[cls.VIDEO_TAR],
                    repo_type="dataset",
                    local_dir=cls.VIDEO_HOME,
                    local_dir_use_symlinks=False,
                )

            log.info(f"[{cls.DATASET_NAME}] Extracting {tar_path}...")
            subprocess.run(
                ["tar", "-xf", tar_path, "-C", cls.VIDEO_HOME,
                 "--checkpoint=1000", "--checkpoint-action=echo=#%u files extracted"],
                check=True,
            )

            cls._check_videos(video_fps_map)



def get_video_source(sources=None) -> Dict[str, Molmo2TrackVideoSource]:
    """Get video source downloader classes by name.

    Args:
        sources: list of source names, or None for all.
    Returns:
        dict of {source_name: downloader_class}
    """
    all_sources = {
        'mose': MOSESource,
        'mosev2': MOSEv2Source,
        'sav': SAVSource,
        'vipseg': VIPSegSource,
        'animaltrack': AnimalTrackSource,
        'APTv2': APTv2Source,
        'bft': BFTSource,
        'soccernet': SoccerNetSource,
        'sportsmot': SportsMOTSource,
        'teamtrack': TeamTrackSource,
        'mot2020': MOT20Source,
        'personpath22': PersonPath22Source,
        'dancetrack': DanceTrackSource,
        'bdd100k': BDD100KSource,
        'uavdt': UAVDTSource,
        'seadrones': SeaDronesSeeSource,

        'youtube': MolmoPointTrackAny,
        'synthetic': MolmoPointTrackSyn,
    }
    if sources is None:
        return all_sources
    result = {}
    for s in sources:
        if s in all_sources:
            result[s] = all_sources[s]
        else:
            log.warning(f"Unknown video source: '{s}'. Available: {list(all_sources.keys())}")
    return result

# ── Standalone Download / Load Helpers ─────────────────────────────────────

DATASET_CLASSES = {
    'train': Molmo2VideoTrack,
    'eval': Molmo2VideoTrackEval,
    'molmopoint-trackany': MolmoPointTrackAny,
    'molmopoint-tracksyn': MolmoPointTrackSyn,
}



def _resolve_sources_from_configs(dataset_cls, configs):
    """For eval dataset, resolve config names to video source names."""
    if not hasattr(dataset_cls, 'CONFIGS'):
        return None
    available_configs = dataset_cls.CONFIGS
    if configs is None:
        configs = list(available_configs.keys())
    sources = []
    for c in configs:
        if c not in available_configs:
            log.warning(f"Unknown config '{c}'. Available: {list(available_configs.keys())}")
            continue
        sources.extend(available_configs[c])
    return sources

def download_video_sources(dataset='train', sources=None, configs=None, n_procs=1):
    """Download and create videos for specified sources.

    Args:
        dataset: 'train', 'eval', 'molmopoint-trackany', or 'molmopoint-tracksyn'
        sources: list of source names (e.g. ['mose', 'dancetrack']), or None for all.
        configs: for eval dataset, config names to resolve to sources.
        n_procs: num workers for parallel video creation.
    """

    if dataset == 'train':
        log.info("Downloading video sources for training dataset...")
        cls = Molmo2VideoTrackInstruction
        cls.download(n_procs=n_procs, sources=sources)

    elif dataset == 'eval':
        log.info("Downloading video sources for eval dataset...")
        cls = Molmo2VideoTrackEval
        # For eval, resolve configs -> sources if no explicit sources given
        if sources is None and configs is not None:
            sources = _resolve_sources_from_configs(cls, configs)
        cls.download(n_procs=n_procs, sources=sources, configs=configs)

    elif dataset == 'molmopoint-trackany':
        log.info("Downloading MolmoPoint-TrackAny dataset...")
        MolmoPointTrackAny.download(n_procs=n_procs)

    elif dataset == 'molmopoint-tracksyn':
        log.info("Downloading MolmoPoint-TrackSyn dataset...")
        MolmoPointTrackSyn.download(n_procs=n_procs)


def list_sources(dataset='train'):
    """Print available video source names."""
    if dataset in ('molmopoint-trackany', 'molmopoint-tracksyn'):
        cls = DATASET_CLASSES[dataset]
        print(f"  {cls.__name__}:")
        print(f"    HF source: {cls.HF_SOURCE}")
        print(f"    Video home: {cls.VIDEO_HOME}")
        if hasattr(cls, 'MANUAL_DOWNLOAD_INSTRUCTION'):
            print(f"    Download instructions:{cls.MANUAL_DOWNLOAD_INSTRUCTION}")
        return

    if dataset == 'train':
        cls = Molmo2VideoTrackInstruction
    else:
        cls = Molmo2VideoTrackEval

    if hasattr(cls, 'CONFIGS'):
        print(f"  Configs ({cls.__name__}):")
        for config, config_sources in cls.CONFIGS.items():
            print(f"    {config}: {config_sources}")
        print()
        # Collect all sources from configs
        all_sources = []
        for s in cls.CONFIGS.values():
            all_sources.extend(s)
        sources = all_sources
    elif hasattr(cls, 'SOURCES'):
        sources = cls.SOURCES
    else:
        sources = None

    if sources:
        for name, dl_cls in get_video_source(sources).items():
            note = f" -- {dl_cls.DOWNLOAD_NOTE}" if dl_cls.DOWNLOAD_NOTE else ""
            print(f"  {name}{note}")
    else:
        for name, dl_cls in get_video_source().items():
            note = f" -- {dl_cls.DOWNLOAD_NOTE}" if dl_cls.DOWNLOAD_NOTE else ""
            print(f"  {name}{note}")


# ── CLI Entry Point ───────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    DATASET_CHOICES = ["train", "eval", "molmopoint-trackany", "molmopoint-tracksyn"]

    parser = argparse.ArgumentParser(
        description="Download Molmo2-VideoTrack annotations and/or video sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Load Molmo2VideoTrack dataset and check the first item
  python -m olmo.data.molmo2_video_track_datasets

  # Load Molmo2VideoTrackEval dataset
  python -m olmo.data.molmo2_video_track_datasets --dataset eval

  # Download videos for specific sources
  python -m olmo.data.molmo2_video_track_datasets --download --sources mose dancetrack

  # Download eval dataset (all configs)
  python -m olmo.data.molmo2_video_track_datasets --download --dataset eval --all

  # Download eval dataset for specific configs
  python -m olmo.data.molmo2_video_track_datasets --download --dataset eval --configs misc animal

  # Download MolmoPoint-TrackAny (YouTube + MammalNet)
  python -m olmo.data.molmo2_video_track_datasets --download --dataset molmopoint-trackany

  # Download MolmoPoint-TrackSyn (synthetic videos)
  python -m olmo.data.molmo2_video_track_datasets --download --dataset molmopoint-tracksyn

  # List info for MolmoPoint-TrackAny
  python -m olmo.data.molmo2_video_track_datasets --list --dataset molmopoint-trackany

"""
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="train",
                        help="Which dataset to operate on (default: train).")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Eval config(s) to download (e.g. misc animal). Only used with --dataset eval.")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Video source(s) to download. Omit for all.")
    parser.add_argument("--list", action="store_true",
                        help="List available video sources and exit.")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets")
    parser.add_argument("--n-procs", type=int, default=8,
                        help="Number of parallel workers (default: 8).")

    args = parser.parse_args()

    if args.list:
        print(f"Available video sources ({args.dataset}):")
        list_sources(dataset=args.dataset)
        sys.exit(0)

    if args.download:
        download_video_sources(dataset=args.dataset, sources=args.sources,
                               configs=args.configs, n_procs=args.n_procs)
    else:
        # Load datasets
        log.info("Loading datasets...")
        if args.dataset == 'train':
            dataset = Molmo2VideoTrack(split='train', task='track', sources=args.sources)
        elif args.dataset == 'eval':
            dataset = Molmo2VideoTrackEval(split='test', task='track', sampling_fps=1, configs=args.configs)
        elif args.dataset in ('molmopoint-trackany', 'molmopoint-tracksyn'):
            cls = DATASET_CLASSES[args.dataset]
            dataset = cls(split='train', task='track')
        log.info(f"Loaded {len(dataset)} videos for {args.dataset} dataset.")
        for ds in tqdm(dataset, desc="Loading dataset", total=len(dataset)):
            pass