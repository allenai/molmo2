"""Class for doing prompting/other data formatting for tasks

For example, converting points to text, or applying prompt templates
"""
import dataclasses
import random
from collections import Counter
import string
from typing import Optional, Dict, Tuple, List, Union, Any

import numpy as np

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.preprocessing.data_formatter import GENERAL_PROMPTS_V1, apply_keyword_prompt, DEMO_STYLES, \
    seconds_to_timestamp, VIDEO_MC_STYLES, IMAGE_MC_STYLES
from olmo.preprocessing.multiple_choice_templates import template_mc_question
from olmo.preprocessing.point_formatter import UnifiedPointFormatter
from olmo.util import parse_timestamp


@dataclasses.dataclass
class Message:
    text: str
    """Text string"""

    points: Union[List, np.ndarray, None] = None
    """Points embedded in the text
    
    The point are coordinates before pre-processing (either [x, y] or [t, x, y]) and
    token IDs (patch_id, subpatch_id, location_id) after pre-processing 
    """


@dataclasses.dataclass
class MolmoPointDataFormatter(BaseConfig):
    """
    Applies prompt templates and adds system prompts to construct text inputs/output

    Includes methods for formatting points, video points, and annotated text.
    Also provides methods for converting points to text and formatting video points.
    """
    prompt_templates: str = "none"  # How to template prompts for examples
    message_format: str = "none"  # How to format messages
    system_prompt: Optional[str] = None  # How to generate system prompts
    always_start_with_space: bool = False  # Always include a leading space for the first bit of text
    default_inference_len: Optional[int] = 65  # Inference len for length-conditioned prompting
    select_answer: str = "best"  # How to select answer for questions with many answers
    debug: bool = False  # deterministic mode for debugging
    eval_system_prompt_mapping: Optional[Dict[str, str]] = None # Custom mapping from eval system prompt styles to train styles
    p_choice_content_in_mc: float = 1.0
    points_decimal_places: int = 1  # Decimal places for points in text
    use_seperate_non_pointing_qa_style: bool = False
    timestamp_mode: str = "50-percent-seconds"
    output_timestamp_mode: str = "seconds"
    seconds_decimal_places: int = 1
    p_multi_point_all_image: float = 0

    use_seperate_count_without_pointing_style: bool = False

    sample_random_initial_point: bool = True  # For video point tracking, whether to sample random initial point
    _point_start_token: bool = False
    include_point_number: Optional[str] = "no_space"

    _end_with_patch: bool = False
    _location_token: bool = False

    def __post_init__(self):
        if self.prompt_templates == "uber_model_v2":
            assert self.system_prompt != "demo_or_style"

    def _get_scale(self, example):
        """Get scale factor from example."""
        if "point_scale" in example:
            # Points are already normalized
            scale = example["point_scale"]
            return [scale, scale]
        elif "image" in example and isinstance(example["image"], np.ndarray):
            h, w = example["image"].shape[:2]
            return [w, h]
        else:
            # Points are in pixel coordinate
            w = example.get("width", example.get("w"))
            h = example.get("height", example.get("h"))
            return [w, h]

    def _select_normalized_human_readable_label(self, example: Dict, rng) -> Tuple[str, str]:
        """Select a normalized label and find its human-readable equivalent."""
        # Get unique normalized labels and select one
        unique_labels = list(set(example["normalized_labels"]))
        selected_normalized_label = rng.choice(unique_labels)

        # Find the original human-readable label for this normalized label
        selected_label = self._find_human_readable_label(example, selected_normalized_label)

        return selected_normalized_label, selected_label

    def _find_human_readable_label(self, example: Dict, normalized_label: str) -> str:
        """Find the original human-readable label for a normalized label."""
        for i, norm_label in enumerate(example["normalized_labels"]):
            if norm_label == normalized_label:
                return example["labels"][i]
        return normalized_label  # fallback

    def template_options(self, example, is_training, rng):
        labelled_options = "options" in example
        allow_unlabelled = True
        if labelled_options and "answer_idx" in example:
            idx = example["answer_idx"]
            if isinstance(idx, int):
                allow_unlabelled = bool(str(example["options"][idx]).strip())

        # If the correct answer is a blank make sure we label the options we there is
        # something sensible to output
        if not is_training or rng.random() < 0.1:
            # "Standard case that we default to during eval
            # We oversample this case to 100% sure its will covered in the training data
            if labelled_options:
                prefixes = string.ascii_uppercase
                options = example["options"]
                option_text = "\n".join(f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options))
                option_names = prefixes[:len(options)]
                outputs = [
                    f"{name}. {opt}" for name, opt in zip(option_names, options)
                ]
            else:
                options = example["unlabelled_options"]
                option_text = "\n".join(options)
                outputs = options
                option_names = options
            question = example["question"] + "\nOnly return the correct answer option.\n" + option_text
        else:
            question = example["question"]

            if labelled_options:
                options = example["options"]
            else:
                options = example["unlabelled_options"]
            question, option_names, outputs = template_mc_question(
                question, options, rng,
                p_label_options=0.8 if allow_unlabelled else 1.0

            )

        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            if not (0 <= ans_idx < len(option_names)):
                raise ValueError(f"Invalid answer idx in example: {example}")
            output = outputs[ans_idx]
        else:
            output = None
        return question, output, dict(option_names=option_names)

    def format_options(self, example):
        if "options" in example:
            prefixes = string.ascii_uppercase
            options = example["options"]
            option_text = "\n".join(f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options))
            option_names = prefixes[:len(options)]
        else:
            options = example["unlabelled_options"]
            option_text = "\n".join(options)
            prefixes = options
            option_names = options
        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            p = random.random()
            if "options" in example and p < self.p_choice_content_in_mc and example.get("content_in_mc", True):
                output = f"{prefixes[ans_idx]}. {options[ans_idx]}"
            else:
                output = prefixes[ans_idx]
        else:
            output = None
        return output, example["question"] + "\n" + option_text + "\n", dict(option_names=option_names)

    def get_point_string(self, label, points, mode, tracking_ids=None):
        if points is None or len(points) == 0:
            return None, "There are none."

        point_str = tokenizer.TOKEN_INDEX_TOKEN + tokenizer.SUBPATCH_INDEX_TOKEN
        if self._location_token:
            point_str += tokenizer.LOCATION_CLS_TOKEN

        if self._end_with_patch:
            count = len(points) - 1
            assert np.all(points[-1, 1:] == -1)
        else:
            count = len(points)
        is_tracking = tracking_ids is not None
        if not is_tracking:
            example_ids = range(1, count + 1)
        else:
            assert len(tracking_ids) == count
            example_ids = tracking_ids

        if self.include_point_number in ["no_space", "True"]:
            point_str = "".join(str(i) + point_str for i in example_ids)
        elif self.include_point_number in ["no_space_id_last"]:
            point_str = "".join(point_str + str(i) for i in example_ids)
        elif self.include_point_number in ["space_id_last"]:
            point_str = "".join(point_str + " " + str(i) for i in example_ids)
        elif self.include_point_number == "with_space":
            point_str = "".join("  " + str(i) + point_str for i in example_ids)
        elif self.include_point_number == "space_only":
            point_str = " ".join(point_str for i in example_ids)
        elif self.include_point_number is None:
            point_str = point_str*count
        else:
            raise NotImplementedError(self.include_point_number)

        if self._end_with_patch:
            if self.include_point_number in ["no_space", "True"]:
                point_str += str(len(example_ids)+1)
                point_str += tokenizer.TOKEN_INDEX_TOKEN
            elif self.include_point_number in ["no_space_id_last", "space_id_last"]:
                point_str += tokenizer.TOKEN_INDEX_TOKEN
            else:
                raise NotImplementedError()

        if self._point_start_token:
            point_str = tokenizer.POINT_PROMPT + point_str

        if is_tracking:
            prefix = "tracks"
        else:
            prefix = "points"
        point_str = f'<{prefix} coords=\"{point_str}\">{label}</point>'
        if mode in ["point_then_count", "point_count"]:
            return points, f"Counting the {point_str} shows a total of {count}."
        elif mode in ["count_then_point", "count_point"]:
            return points, f"There are {count} {point_str}."
        elif mode == "count":
            return None, str(count)
        elif mode is None or mode in ["point", "pointing"]:
            return points, point_str
        else:
            raise NotImplementedError(mode)

    def format_points(self, example, points_to_indices):
        points = example["points"]
        count = len(points)
        style = example["style"]
        if "label" in example:
            label = example["label"].lower()
        elif "label_cased" in example:
            label = example["label_cased"]
        else:
            label = example["question"]
        if count > 0:
            scale = self._get_scale(example)
            scale = np.array(scale)[None, :]
            if example.get("clip_points"):
                points = np.clip(points, 0, scale)
            points = points_to_indices(points / scale)
        else:
            points = None
        mode = style
        if style == "cosyn_point":
            mode = "point"
        return self.get_point_string(label, points, mode)

    def format_video_points(self, example, points_to_indices):
        if "points" not in example or "timestamps" not in example:
            raise ValueError("No points provided")
        if "count" in example and example["count"] == 0:
            return None, "There are none."
        if "unanswerable" in example and example["unanswerable"]:
            return None, example["explanation"] if len(example["explanation"]) > 0 else f"Sorry, I can't count {example['label']}."
        all_points = example["points"]
        all_timestamps = example["timestamps"]
        count = sum(len(x) for x in all_points)
        if count == 0:
            point_arr = None
        else:
            point_arr = []
            for ts, points in zip(all_timestamps, all_points):
                for point in points:
                    point_arr.append([ts, point["x"]/100, point["y"]/100])
            point_arr = points_to_indices(np.array(point_arr))
        style = example["style"]
        assert style.startswith("video_")
        mode = style[6:]
        return self.get_point_string(example["label"], point_arr, mode)

    def format_multi_image_points(self, example, points_to_indices):
        """
        Format video points for counting objects across frames.
        """
        style = example["style"]
        assert style.startswith("multi_image_")
        mode = style[len("multi_image_"):]
        points = example["points"]
        if len(points) > 0:
            points = np.array(points, dtype=np.float64)
            if "point_scale" in example:
                scale = example["point_scale"]
                points[:, 1:] /= scale
            else:
                for i, (img, x, y) in enumerate(points):
                    h, w = example["image"][int(img)].shape[:2]
                    points[i, 1:] = (x/w, y/h)
            points = points_to_indices(points)
        return self.get_point_string(example["label"], points, mode)

    def generate_multipoint_query(self, example, point_to_indices, rng):
        """Format multi-image pointing and counting.

        Generates questions and answers for pointing to and counting objects across multiple images.
        Supports various modes: pointing only, counting only, or combined operations.
        """

        # Just to match the RNG state the VideoMolmo DataFormatter
        rng.choice([""], p=[1])

        # Get unique normalized labels and select one
        selected_normalized_label, selected_label = self._select_normalized_human_readable_label(example, rng)

        # about 10% of the time select a negative label.
        # if rng.random() < 0.1:
        #     selected_normalized_label = rng.choice(NEGATIVE_LABELS)
        #     selected_label = selected_normalized_label  # negative labels are already human-readable

        # Use ALL images in the example, not just valid ones
        all_images = list(range(len(example["normalized_labels"])))
        n_images = len(all_images)

        # Check if the selected label exists in any of the images
        label_exists = False
        for i, (label, points) in enumerate(zip(example["normalized_labels"], example["points"])):
            if label == selected_normalized_label and len(points) > 0:
                label_exists = True
                break

        if self.p_multi_point_all_image:
            if n_images == 1 or rng.random() < self.p_multi_point_all_image:
                selected_images = "all images"
            else:
                n_images_to_select = rng.randint(1, n_images)
                selected_images = rng.choice(all_images, size=n_images_to_select)
                selected_images = ", ".join([f"image_{i+1}" for i in selected_images])
        else:
            # Randomly select from 1 to total number of available images plus "all images" option
            n_images_to_select = rng.randint(1, n_images) if n_images >= 1 else n_images
            selected_images = rng.choice(all_images, size=n_images_to_select)

            # Randomly select between "all images" or specific image list
            if n_images_to_select == n_images and rng.random() < 0.5:
                selected_images = "all images"
            else:
                selected_images = ", ".join([f"image_{i+1}" for i in selected_images])

        style = example.get("style")
        if selected_images == "all images" and style == "multi_image_pointing":
            # 50% chance to use original pointing template for all images
            if rng.random() < 0.5:
                prompt_template = rng.choice(GENERAL_PROMPTS_V1["pointing"])
                question = prompt_template.format(label=selected_label)
            else:
                prompt_template = rng.choice(GENERAL_PROMPTS_V1[style])
                question = prompt_template.format(
                    selected_images=selected_images,
                    selected_label=selected_label
                )

            # NOTE: when "all images", we can always consider Qs without selected images.
        else:
            prompt_template = rng.choice(GENERAL_PROMPTS_V1[style])
            question = prompt_template.format(
                selected_images=selected_images,
                selected_label=selected_label
            )

        # Determine mode based on style
        if style == "multi_image_pointing":
            mode = "point"
        elif style == "multi_image_counting":
            mode = "count"
        elif style == "multi_image_point_then_count":
            mode = "point_then_count"
        elif style == "multi_image_count_then_point":
            mode = "count_then_point"
        else:
            raise NotImplementedError(style)

        if not label_exists:
            points = None
        else:
            # Find images that have this label with points for the answer
            # Only consider the selected images, not all images
            if selected_images == "all images":
                valid_images = []
                for i, (label, points) in enumerate(zip
                                                        (example["normalized_labels"], example["points"])):
                    if label == selected_normalized_label and len(points) > 0:
                        valid_images.append(i)
            else:
                # Convert "image_1, image_2" to [0, 1]
                selected_indices = [int(img.split('_')[1]) - 1
                                    for img in selected_images.split(', ')]
                valid_images = []
                for i in selected_indices:
                    if (i < len(example["normalized_labels"]) and
                        example["normalized_labels"][i] == selected_normalized_label and
                        len(example["points"][i]) > 0):
                        valid_images.append(i)

            if not valid_images:
                points = None
            else:
                # Prep for format_multi_image_points
                image_indices = []
                points_list = []
                for i in valid_images:
                    points = example["points"][i]
                    for point in points:
                        if example.get("clip_points"):
                            scale = example['point_scale']
                            if isinstance(point, dict) and 'x' in point and 'y' in point:
                                x, y = max(0, min(point['x'], scale)), max(0, min(point['y'], scale))
                                points_list.append((i, x/scale, y/scale))
                            elif isinstance(point, (list, tuple)) and len(point) == 2:
                                x, y = max(0, min(point[0], scale)), max(0, min(point[1], scale))
                                points_list.append((i, x/scale, y/scale))
                        else:
                            points_list.append((i, point[0], point[1]))
                points = point_to_indices(np.array(points_list))
        points, answer = self.get_point_string(selected_label, points, mode)
        return points, question, answer

    def format_video_input_points(self, initial_points, scale) -> str:
        raise NotImplementedError()

    def _find_initial_points(self, frames_data) -> list:
        """
        Find initial points from their first visible frame.
        Used for findinig inital query points to track for point tracking

        Returns:
            List of dicts with keys: id, point [x,y], time, frame
        """
        initial_points = {}
        for frame_data in frames_data:
            for point_id, point_info in frame_data["points"].items():
                if point_id not in initial_points:
                    if not point_info.get("occluded", False):
                        initial_points[point_id] = {
                            'id': point_id,
                            'point': point_info["point"],
                            'time': frame_data["time"],
                            'frame': frame_data["frame"]
                        }
        initial_points = list(initial_points.values())
        initial_points.sort(key=lambda x: x['frame']*10000 + x['point'][0]*100 + x['point'][1]) # Sort by frame, then x, then y

        return initial_points

    def _sample_initial_point(self, frames_data, input_point_id, is_training, rng):
        """
        Sample a single initial point from the first visible frame for the given point_id.
        Used for finding initial query point to track for single point tracking.
        If training, sample randomly among visible points.

        Returns:
            Dict with keys: id, point [x,y], time, frame
        """

        # Sample randomly among visible points
        visible_points = []
        for frame_data in frames_data:
            for point_id, point_info in frame_data["points"].items():
                if point_id == input_point_id:
                    if not point_info.get("occluded", False):
                        visible_points.append({
                            'id': point_id,
                            'point': point_info["point"],
                            'time': frame_data["time"],
                            'frame': frame_data["frame"]
                        })

        if not visible_points:
            return None

        if is_training and self.sample_random_initial_point: # sample beginning or any visible point
            return rng.choice(visible_points) if rng.random() < 0.2 else visible_points[0]
        else: # pick the first visible point for eval
            return visible_points[0]
    def _filter_frames_to_video(self, frames_data, video_timestamps, eps=1e-2):
        """
        Filter frames_data to only include frames that match actual video timestamps.
        Uses numpy broadcasting for efficient comparison.
        """
        if not frames_data or video_timestamps is None or len(video_timestamps) == 0:
            return []

        # Extract frame times from frames_data
        frame_times = np.array([parse_timestamp(f["time"]) for f in frames_data])

        # Compute difference matrix: (n_frames, n_video_timestamps)
        video_timestamps = np.array(video_timestamps)
        diff_matrix = np.abs(frame_times[:, None] - video_timestamps)

        # Find minimum difference for each frame and corresponding video index
        min_diffs = np.min(diff_matrix, axis=1)
        closest_indices = np.argmin(diff_matrix, axis=1)

        # Filter frames that have a match within epsilon
        filtered_frames = []
        for i, frame_data in enumerate(frames_data):
            if min_diffs[i] < eps:
                filtered_frame = dict(frame_data)
                # filtered_frame["frame"] = int(closest_indices[i])
                # filtered_frame["time"] = float(video_timestamps[closest_indices[i]])
                filtered_frames.append(filtered_frame)

        return filtered_frames

    def _sample_at_fps(self, frames_data, sampling_fps):
        """
        Sample frames at specified fps interval.
        Trick: generate timestamp grids based on sampling_fps, then use _filter_frames_to_video to algin frames.
        """
        if not frames_data or sampling_fps <= 0:
            return frames_data

        sampling_interval = 1.0 / sampling_fps

        # Generate target times on the sampling grid
        start_time = parse_timestamp(frames_data[0]["time"])
        end_time = parse_timestamp(frames_data[-1]["time"])

        # Align to grid: find first grid point >= start_time
        first_grid_point = np.ceil(start_time / sampling_interval) * sampling_interval

        # Generate grid points
        target_times = np.arange(first_grid_point, end_time + 1e-6, sampling_interval)

        # Use filter_frames_to_video to find closest frames to these target times
        return self._filter_frames_to_video(frames_data, target_times)

    def format_video_object_track_points(self, example, is_training, rng, points_to_indices):
        """
        Format video points for tracking objects across frames.
        Keep only frames that match actual video timestamps.
        Sample frames at specified sampling fps for per-frame tracking.
        """
        style = example["style"]
        label = example["label"]
        sampling_fps = example["sampling_fps"]
        input_points = None
        scale = self._get_scale(example)

        if "points" not in example or not example["points"]:
            prompt_keywords = dict(label=label)
            if sampling_fps and sampling_fps > 0:
                prompt_keywords["fps"] = str(int(sampling_fps))
            prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], prompt_keywords, rng, dbg=self.debug)
            return None, prompt, "There are none."

        # Get actual video timestamps if available
        video_info = example.get("video", {})
        timestamps = video_info.get("timestamps", None)

        # Filter frames to match actual video timestamps
        frames_data = example["points"]
        frames_data = self._filter_frames_to_video(frames_data, timestamps)
        # NOTE: if frames_data is empty after filtering, we still proceed to sample initial points from original frames later
        # Output will be just "There are none." in that case

        # Get ouptut
        if style == "video_point_track_per_frame":
            # Apply fps sampling if specified
            if sampling_fps and sampling_fps > 0:
                frames_data = self._sample_at_fps(frames_data, sampling_fps)
            point, output = self.format_video_tracks(
                frames_data, scale, label, points_to_indices)


        elif style == "video_point_ground_start_end":
            # For ground_start_end, just use filtered frames without fps sampling
            point, output = self.format_video_tracks(
                frames_data, scale, label, points_to_indices, start_end_only=True)

        elif style == "video_single_point_track_per_frame":
            # Sample intial point from first visible frame or randomly if training
            initial_points = example.get("initial_points")
            if not initial_points:
                # For single point tracking, we assume tracking only one point with id=0
                initial_point = self._sample_initial_point(frames_data, input_point_id=0, is_training=is_training, rng=rng)
                if initial_point is None: # no visible initial point found after filtering, so randomly pick one from original frames
                    print("No visible initial point found after filtering frames to video timestamps for single point tracking, sampling from original frames instead.")
                    initial_point = self._sample_initial_point(example["points"], input_point_id=0, is_training=is_training, rng=rng)
                assert initial_point is not None, "No visible initial point found for single point tracking"
                initial_points = [initial_point]

            sampling_fps = example.get("sampling_fps")
            if sampling_fps and sampling_fps > 0:
                frames_data = self._sample_at_fps(frames_data, sampling_fps)

            # Design prompt for input points
            point, output = self.format_video_tracks(
                frames_data, scale, label,
                single_point_track=True,
                from_initial_points=initial_points,
                points_to_indices=points_to_indices,
            )
            # Not really clear how to handle this for token indexing, for now we just
            # use text coordiantes for the input point
            assert sorted(x["id"] for x in initial_points) == list(range(len(initial_points)))
            input_points = UnifiedPointFormatter().format_video_points(
                [x['time'] for x in initial_points],
                [[x['point']] for x in initial_points],
                scale,
                label=example["label"],
                mode=None
            )
        else:
            raise NotImplementedError(f"Unsupported video point style: {style}")


        # assert len(frames_data) > 0, "No frames left after filtering/sampling"
        if False and "question" in example:
            prompt = example["question"]
        else:
            prompt_keywords = dict(label=label)
            if sampling_fps and sampling_fps > 0:
                prompt_keywords["fps"] = str(int(sampling_fps))
            if input_points is not None:
                prompt_keywords["input_points"] = input_points
            if style == "video_point_track_per_frame" and prompt_keywords["fps"] == '2' and rng.random() < 0.5:
                del prompt_keywords["fps"]
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["video_point_track_per_frame_default_fps"], prompt_keywords, rng, dbg=self.debug)
            else:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], prompt_keywords, rng, dbg=self.debug)
        return point, prompt, output

    def format_video_point_track_points(self, example, initial_points):
        """
        Format video points for tracking points across frames

        Input:
            example: dict with keys: style, points (list of dicts with keys: frame, time, points)
            initial_points: dict of initial points with keys: id, point [x,y], frame

        Output formats by style:
        - video_point_track_all_frames_with_occlusion: "time {t}\n{id: [x, y, occluded], ...}"
        """
        if "points" not in example or not example["points"]:
            return "There are none."

        style = example["style"]
        frames_data = example["points"]
        scale = self._get_scale(example)
        return self.format_video_tracks(frames_data, scale, example["label"], from_initial_points=initial_points)

    def _scale_point(self, point, scale):
        if isinstance(scale, (tuple, list)):
            x_scale, y_scale = scale
        else:
            x_scale, y_scale = scale, scale
        x, y = float(point[0])/x_scale, float(point[1])/y_scale
        return (max(0, min(x, 1.0)), max(0, min(y, 1.0)))

    def format_video_tracks(self, frames_data, scale, label, points_to_indices, alt_text=None, rng=None,
                            start_end_only=False, single_point_track=False,
                            from_initial_points=None):
        if len(frames_data) == 0:
            return None, "No tracks available."

        if start_end_only:
            frames_data = UnifiedPointFormatter._filter_all_but_start_end(frames_data)
        if from_initial_points is not None:
            frames_data = UnifiedPointFormatter._filter_for_initial_points(frames_data, from_initial_points)

        points = []
        example_ids = []
        for frame_data in frames_data:
            if not frame_data["points"]:
                continue
            points_dict = frame_data["points"]
            for obj_id, point_info in points_dict.items():
                occluded = point_info.get("occluded", False)
                if occluded:
                    continue
                x, y = self._scale_point(point_info["point"], scale)
                time = frame_data["time"]
                points.append((frame_data["time"], x, y))
                example_ids.append(obj_id)

        points, example_ids = points_to_indices(points, example_ids)
        return self.get_point_string(label, points, "point", example_ids)

    def select_vqa_answer(self, answers, rng):
        if answers is None or isinstance(answers, str):
            return answers
        if self.select_answer == "first":
            return min(answers)
        if self.select_answer == "best":
            counts = Counter(answers)
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            return candidates[rng.randint(0, len(candidates))]
        else:
            raise NotImplementedError(self.select_answer)

    def format_messages(self, messages: List[Message]) -> List[Message]:
        """Applies system formatting to ith message from a sequence of messages"""
        for ix, message in enumerate(messages):
            is_user = ix % 2 == 0
            if self.message_format == "qwen3":
                if is_user:
                    if ix != 0:
                        prefix = "<|im_end|>\n"
                    else:
                        prefix = ""
                    message.text = f"{prefix}<|im_start|>user\n{message.text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if self.message_format == "none" or self.message_format is None:
                    pass
                elif self.message_format == "role":
                    if is_user:
                        message.text = "User: " + message.text + " Assistant:"
                else:
                    raise NotImplementedError(self.message_format)

                if ix != 0 or self.always_start_with_space:
                    message.text = " " + message.text
        return messages

    def get_system_prompt(self, style, for_inference, messages, is_training, rng):
        # For eval only dataset
        if self.eval_system_prompt_mapping is not None and style in self.eval_system_prompt_mapping:
            # Prioritize eval system prompt mapping if provided
            style = self.eval_system_prompt_mapping[style]
        else:
            # For eval, or if use_seperate_non_pointing_qa_style has been turned off,
            # remove the "count_without_pointing" style
            if (not is_training) or (not self.use_seperate_count_without_pointing_style):
                if style == "video_short_answer_count_without_pointing":
                    style = "video_short_answer"
                if style == "video_multiple_choice_count_without_pointing":
                    style = "video_multiple_choice"

            if style == "eval_short_answer":
                style = "vqa2"
            elif style == "eval_multiple_choice":
                style = "a_okvqa_mc"
            elif style == "video_eval_short_answer":
                style = "video_short_answer"
            elif style == "video_eval_multiple_choice":
                style = "video_multiple_choice"
            elif style == "video_eval_multiple_choice_w_subtitle":
                style = "video_multiple_choice_w_subtitle"
            elif style.startswith("eval_multi_image_"):
                style = style[len("eval_"):]

        if self.system_prompt == "style":
            return style + ":"

        elif self.system_prompt == "demo_or_style":
            if style == "android_control" or style == "demo":
                # android is a special case since I hacked in prefix in the preprocessor
                prefix = ""
            elif style in DEMO_STYLES and rng.random() > 0.1 and not self.debug:
                # Use style prompt 10% of the time so we can still get task-specific output
                prefix = ""
            else:
                prefix = style + ":"

        elif self.system_prompt == "demo_or_style_v2":
            # not percent chance to style use the style tag, all MC questions do not get a
            # style tag
            if style in DEMO_STYLES or style in VIDEO_MC_STYLES or style in IMAGE_MC_STYLES:
                prefix = ""
            else:
                prefix = style + ":"

        elif self.system_prompt in ["style_and_length", "style_and_length_v2"] and (
            style in ["pointing", "point_count", "cosyn_point", "text_sft",
                      "video_point", "video_point_count", "video_count", "video_count_point",]):
            prefix = style + ":"

        elif for_inference and self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            v2 = self.system_prompt == "style_and_length_v2"
            inference_len = self.default_inference_len
            n = None if inference_len is None else str(inference_len)
            if n is not None and len(n) > 0:  # allow empty string to signal unconditioned
                prefix = style + " " + n + ":"
            else:
                if self.system_prompt in ["style_and_length_v2"]:
                    prefix = style + ":"
                else:
                    prefix = style + " :"
        elif self.system_prompt in ["style_and_length_v3"]:
            # Length hint noise based on a percent of the total length instead of a staticly
            # defined factor
            if for_inference:
                n = self.default_inference_len
            elif rng.random() > 0.10:
                n = len(messages[-1])
                n *= np.clip(rng.normal(scale=0.05, loc=1), 0.5, 1.5)
                n = int(n / 25)
            else:
                n = None
            if n is not None:
                prefix = style + " " + str(n) + ":"
            else:
                prefix = style + ":"
        elif self.system_prompt in ["style_and_length", "style_and_length_v2"]:
            std = 25
            if rng.random() > 0.10:
                if isinstance(messages[-1], str):
                    n = len(messages[-1])
                else:
                    n = len(messages[-1].text)
                n += int(rng.normal(scale=std))
                n = n // 15
            else:
                n = None
            if n is not None:
                prefix = style + " " + str(n) + ":"
            else:
                if self.system_prompt in ["style_and_length_v2"]:
                    prefix = style + ":"
                else:
                    prefix = style + " :"
        elif self.system_prompt == "no_style":
            prefix = ""
        else:
            raise NotImplementedError(self.system_prompt)

        return prefix

    def format_input_timestamps(self, rng, timestamps):
        """Format input timestamp as text"""
        timestamps = [parse_timestamp(x) for x in timestamps]
        if self.timestamp_mode == "rng-v1":
            raise NotImplementedError()
        if self.timestamp_mode == "50-percent-seconds":
            if rng.random() > 0.5:
                return True, [str(round(x, self.seconds_decimal_places)) for x in timestamps]
            else:
                return False, [seconds_to_timestamp(x, self.seconds_decimal_places) for x in timestamps]
        elif self.timestamp_mode == "seconds-to-tenth":
            return True, [str(round(x, 1)) for x in timestamps]
        elif self.timestamp_mode == "seconds":
            return True, [str(round(x, self.seconds_decimal_places)) for x in timestamps]
        else:
            raise NotImplementedError()

    def format_output_timestamp(self, time_value):
        """Format output timestamp as text

        For output timestamps, the model should always use a consistent format
        """
        time_value = parse_timestamp(time_value)
        if isinstance(time_value, str):
            return time_value
        else:
            if self.output_timestamp_mode == "timestamp":
                return seconds_to_timestamp(time_value, self.seconds_decimal_places)
            elif self.output_timestamp_mode == "seconds":
                return str(round(time_value, self.seconds_decimal_places))
            else:
                raise NotImplementedError()

    def get_user_prompt(self, example, is_training=True, for_inference=False, rng=None,
                        points_to_indices=None):
        """Build a list of strings of what a user might type in to the model for the given example,
        and its responses, by applying a prompt template to the fields in `example`

        Uses the `style` field to understand what the task/output style is
        """
        video_object_track_styles = [
            "video_point_track_per_frame", "video_point_ground_start_end", "video_single_point_track_per_frame",
        ]
        video_point_track_styles = [
            "video_point_track_per_frame_with_occlusion",
            "video_point_track_all_frames_with_occlusion"
        ]
        multi_image_pointing_styles = [
            "multi_image_pointing",
            "multi_image_counting",
            "multi_image_point_then_count",
            "multi_image_count_then_point",
        ]

        style = example.get("style")
        output = None
        metadata = None
        points = None
        if "prompt" in example:
            # Examples have a complete user prompt pre-specified, usually for eval sets
            prompt = example["prompt"]

        elif self.prompt_templates == "none":
            # Bare-bone prompt with no templating or instructions
            if "prompt" in example:
                prompt = example["prompt"]
            elif style in ["pointing", "point_count", "point_then_count", "cosyn_point"]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    if "label" in example:
                        prompt = example["label"]
                        prompt = prompt.lower()
                    else:
                        prompt = example["label_cased"]
                if "points" in example:
                    points, output = self.format_points(example, points_to_indices)
            elif "question" in example and ("options" in example or "unlabelled_options" in example):
                output, prompt, metadata = self.format_options(example)
            elif "timestamp" in example:
                prompt = str(round(example["timestamp"], 2))
            elif "start_time" in example:
                prompt = str(round(example["start_time"], 2)) + "-" + str(round(example["end_time"], 2))
            elif "question" in example:
                prompt = example["question"]
            elif style in [
                "video_point",
                "video_point_count",
                "video_count",
                "video_count_point",
            ]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    prompt = example["label"]
                points, output = self.format_video_points(example, points_to_indices)
                metadata = example.get("metadata", {})
                metadata["answer"] = output
            else:
                prompt = ""

        elif self.prompt_templates in ["uber_model", "uber_model_v2"]:
            if self.prompt_templates == "uber_model_v2":
                template_all_multiple_choice = True
            else:
                template_all_multiple_choice = False

            # We template long captions and pointing since they are "demo" tasks, and use
            # plain text for everything else
            if style in [
                "long_caption",
                "short_caption",
                "video_long_caption",
                "video_short_caption",
                "video_transcript",
                "video_motion_caption",
                "video_object_caption",
            ] and "question" not in example:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
            elif style == "video_frame_caption_timestamp":
                in_seconds, (timestamp_str,) = self.format_input_timestamps(rng, [example["timestamp"]])
                prompt = apply_keyword_prompt(
                    GENERAL_PROMPTS_V1[style + ("_in_seconds" if in_seconds else "")],
                    dict(example, timestamp=timestamp_str),
                    rng, dbg=self.debug)
            elif style in ["video_clip_caption_start_end", "video_clip_transcript_start_end"]:
                in_seconds, (start_str, end_str) = self.format_input_timestamps(
                    rng, [example["start_time"], example["end_time"]])
                prompt = apply_keyword_prompt(
                    GENERAL_PROMPTS_V1[style + ("_in_seconds" if in_seconds else "")],
                    dict(example, start_time=start_str, end_time=end_str),
                    rng, dbg=self.debug)
            elif style in ["video_short_answer", "video_short_answer_count_without_pointing"]:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
            elif "_exp" in style:
                prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["chain_of_thought"], example, rng, dbg=self.debug)
            elif style in ["pointing", "point_count", "point_then_count", "cosyn_point"]:
                w_scale, h_scale = self._get_scale(example)
                if "question" in example:
                    prompt = example["question"]
                else:
                    if "label" in example:
                        prompt = example["label"].lower()
                    else:
                        prompt = example["label_cased"]
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], dict(example, label=prompt), rng, dbg=self.debug)
                if "points" in example:
                    points, output = self.format_points(example, points_to_indices)
            elif style in video_object_track_styles:
                points, prompt, output = self.format_video_object_track_points(
                    example, is_training, rng, points_to_indices)
            elif style in video_point_track_styles:
                raise NotImplementedError()
            elif style in [
                "video_point",
                "video_point_count",
                "video_count",
                "video_count_point",
            ]:
                if "question" in example:
                    prompt = example["question"]
                else:
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], example, rng, dbg=self.debug)
                points, output = self.format_video_points(example, points_to_indices)
                metadata = example.get("metadata", {})
                metadata["answer"] = output
            elif style in multi_image_pointing_styles:
                if "normalized_labels" in example:
                    points, prompt, output = self.generate_multipoint_query(example, points_to_indices, rng)
                else:
                    prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["pointing"], example, rng, dbg=self.debug)
                    points, output = self.format_multi_image_points(example, points_to_indices)
                metadata = example.get("metadata", {})
                metadata["answer"] = output
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "question" in example and ("options" in example or "unlabelled_options" in example):
                if template_all_multiple_choice:
                    prompt, output, metadata = self.template_options(example, is_training, rng)
                else:
                    output, prompt, metadata = self.format_options(example)
                    if style in VIDEO_MC_STYLES:
                        # apply template on top of formatted options
                        options_text = prompt.replace(example["question"], "").strip()
                        prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1[style], dict(example, question=example["question"], options=options_text), rng, dbg=self.debug)
            elif "question" in example:
                prompt = example["question"]
            else:
                prompt = ""
        else:
            raise NotImplementedError(self.prompt_templates)

        if output is None and not for_inference:
            if "answers" in example:
                output = self.select_vqa_answer(example["answers"], rng)
            elif "answer" in example:
                output = example["answer"]
                if "answer_annotations" in example:
                    raise NotImplementedError()
                elif "explanation" in example:
                    output = example["explanation"] + " Answer: " + output
            elif "answer_with_points" in example:
                output = example["answer_with_points"]
            elif "text" in example:
                output = example["text"]
            else:
                raise ValueError("No output in example, if this is an inference-only task make sure `for_inference` is True")

        return Message(prompt), Message(output, points), metadata

    def _format_example(self, message, example, is_training, for_inference, rng, points_to_indices):
        metadata = {}
        messages: List[Message]
        for k in ["answer_idx", "answers", "answer", "points", "options"]:
            if k in message:
                metadata[k] = message[k]
        if isinstance(message, str):
            messages = [Message(message)]
        elif isinstance(message, list):
            messages = [Message(x) for x in message]
        elif "messages" in message:
            # Example directly contains the prompts/message to use
            messages = [Message(x) for x in message["messages"]]
        elif isinstance(message, dict):
            # An example that requires a custom prompt
            if "video" in example:
                video = example["video"]
                if hasattr(video, "timestamps"):  # JAMES: use loaded video to sample point tracks with aligned timestamps and fps
                    message["video"] = {"timestamps": video.timestamps, "target_fps": video.target_fps}
            if "image" in example:
                message["image"] = example["image"]
            if "multi_turn_messages" in example:
                messages = []
                # multi-turn conversations that needs to be formatted through `get_user_prompt`
                for turn_message in message["multi_turn_messages"]:
                    prompt, response, extra_metadata = self.get_user_prompt(
                        turn_message, is_training, for_inference=for_inference, rng=rng,
                        points_to_indices=points_to_indices
                    )
                    assert response is not None
                    messages += [prompt, response]
            else:
                prompt, response, extra_metadata = self.get_user_prompt(
                    message, is_training, for_inference=for_inference, rng=rng,
                    points_to_indices=points_to_indices
                )
                if extra_metadata:
                    metadata.update(extra_metadata)
                if not for_inference:
                    assert response is not None
                    messages = [prompt, response]
                else:
                    messages = [prompt]
        else:
            raise ValueError(f"Example type {type(message)} not understood")

        # Add the system prompt
        if self.system_prompt and self.system_prompt != "none":
            style = None
            if isinstance(message, dict):
                if "multi_turn_messages" in message:
                    # FIXME This is a bit of hack, its okay for now since our only multi-turn
                    # messages use the "demo" style
                    style = message["multi_turn_messages"][0]["style"]
                else:
                    style = message.get("style", None)
            prefix = self.get_system_prompt(style, for_inference, messages, is_training, rng=rng)
            if len(prefix) > 0:
                if len(messages[0].text) > 0:
                    messages[0].text = prefix + " " + messages[0].text
                else:
                    messages[0].text = prefix

        # Add the role annotations such as "User:" and "Assistant:"
        messages = self.format_messages(messages)
        return messages, metadata

    def __call__(self, ex: Dict, is_training, for_inference, rng, points_to_indices) -> Tuple[Dict, Dict]:
        """Returns a formatted example and example metadata"""
        if "message_list" in ex:
            # Does not support returning metadata, which is fine since we are not doing inference
            return [self._format_example(msg, ex, is_training, for_inference, rng, points_to_indices)[0]
                    for msg in ex["message_list"]], None
        elif "messages" in ex:
            return self._format_example(ex["messages"], ex, is_training, for_inference, rng, points_to_indices)
        else:
            return self._format_example(ex, ex, is_training, for_inference, rng, points_to_indices)