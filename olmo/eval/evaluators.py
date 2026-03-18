"""Classes the compute metrics given ground truth/prediction pairs"""
import base64
import dataclasses
import io
import json
import logging
import os
import re
import copy
import math
import random
import string

from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape as html_escape
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchmetrics import MeanMetric, SumMetric, Metric

from .tomato_eval import get_tomato_score
from .llm_judge_utils import llm_judge_score
from .vqa import vqa_score, anls_metric, relaxed_correctness, scifi_relaxed_correctness, \
    a_okvqa_score, select_mc_option, mmmu_score, real_world_qa_score, math_vista_score, \
    select_perception_test_option, select_ego_schema_option, nextqa_mc, muir_bench_mc, \
    mantis_eval_mc, ego3d_bench_score
from .vsi_bench_utils import vsi_bench_na_score
from .temp_compass_utils import temp_compass_score
from .mlvu_utils import mlvu_ssc_score, mlvu_summary_score
from .video_caption_utils import eval_caption
from .vixmo_caption_utils import eval_vixmo_caption, eval_vixmo_caption2
from .dream_caption_utils import eval_dream_caption
from .object_tracking_utils import evaluate_video_object_tracking
from ..html_utils import build_html_table, postprocess_prompt, BoxesToVisualize, \
    get_html_image_with_boxes, get_image_collage_coords_from_video_points, get_frame_coordinates_in_collage
from ..io import write_file
from ..preprocessing.point_formatter import extract_multi_image_points, extract_tracks, \
    extract_points
from ..torch_util import (
    get_global_rank,
    get_world_size, barrier, get_local_world_size,
    gather_object
)
from ..util import flatten_list, interpolate_frame_scores, \
    normalize_timestamps_and_points
from ..util import flatten_list, interpolate_frame_scores, \
    normalize_timestamps_and_points

log = logging.getLogger(__name__)


def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return key


def mean_metric(v):
    metric = MeanMetric(nan_strategy="error")
    metric.update(np.mean(v) if len(v)>0 else 0, len(v))
    return metric


def sum_metric(v):
    metric = SumMetric(nan_strategy="error")
    metric.update(np.sum(v) if len(v)>0 else 0)
    return metric


def _extract_image_points(predictions, ix, image_w, image_h):
    if "points" in predictions:
        points = predictions["points"][ix]
        if len(points) > 0:
            points = points[:, 1:]
    else:
        text = predictions["predictions_text"][ix]
        points = extract_points(text, image_w, image_h)
    return points


@dataclasses.dataclass
class HtmlTable:
    """Returned as special metric for visualizing predictions"""
    rows: List[Dict[str, Any]]

    def get_html(self):
        return build_html_table(self.rows)


def annotation_to_box(points, point_dist=4):
    to_show = []
    for point in points:
        if len(point) == 2:
            x, y = point
            to_show.append([x-point_dist, y-point_dist, x+point_dist, y+point_dist])
        else:
            to_show.append(point)
    return to_show


def gather_examples_as_html(
    n_examples, voc, metadatas, predictions,
    scores=None, fix_width=True, pred_points=None, gt_points=None,
    pred_bboxes=None, gt_bboxes=None,
    pred_times_and_points=None, gt_times_and_points=None,
) -> HtmlTable:
    """Builds a HTML table visualization of the predictions"""

    n = len(predictions["predictions"])
    if n_examples is not None:
        # Divide by world size since we will aggregate visualization across all processes
        n = min(n, n_examples)
        n = (n + get_world_size() - 1) // get_world_size()
    rows = []
    new_tokens = predictions["predictions"]
    prompt_tokens = predictions["prompts"]
    bmm = predictions["bmm"] if "bmm" in predictions else None
    high_res_indices = predictions["high_res_indices"] if "high_res_indices" in predictions else None
    for ix in range(n):
        prompt_text = postprocess_prompt(voc.decode(prompt_tokens[ix][prompt_tokens[ix] >= 0]))
        metadata = metadatas[ix]
        pred_seq = new_tokens[ix]
        pred_txt = voc.decode(pred_seq[pred_seq >= 0])

        image_src = None
        if "image_url" in metadata:
            image_src = metadata['image_url']
        elif "image" in metadata and isinstance(metadata["image"], np.ndarray):
            img = Image.fromarray(metadata["image"])
            if high_res_indices is not None:
                img_w, img_h = 128, 128
                num_cols = int(img.size[0] / img_w)

                annotate_boxes = high_res_indices[ix]
                draw = ImageDraw.Draw(img)
                for annotate_index in annotate_boxes:
                    if annotate_index == -1:
                        continue
                    annotate_index = int(annotate_index)
                    row_idx = annotate_index // num_cols
                    col_idx = annotate_index % num_cols

                    x1 = col_idx * img_w
                    y1 = row_idx * img_h
                    x2 = x1 + img_w - 1
                    y2 = y1 + img_h - 1

                    # Draw the rectangle with desired thickness
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

            image_data = io.BytesIO()
            img.save(image_data, format='JPEG')
            image_data = image_data.getvalue()
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'
        elif "image" in metadata:
            with Image.open(metadata["image"]) as img:
                image_data = io.BytesIO()
                img.save(image_data, format='JPEG')
                image_data = image_data.getvalue()
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'

        row = dict()
        if image_src is not None:
            ex_pred_points, gt_pred_points, ex_pred_bboxes, ex_gt_bboxes = None, None, None, None
            ex_pred_times_and_points, ex_gt_times_and_points = None, None
            if pred_points is not None:
                ex_pred_points = pred_points[ix]
            if gt_points is not None:
                gt_pred_points = gt_points[ix]
            if pred_bboxes is not None:
                ex_pred_bboxes = pred_bboxes[ix]
            if gt_bboxes is not None:
                ex_gt_bboxes = gt_bboxes[ix]
            if pred_times_and_points is not None:
                ex_pred_times_and_points = pred_times_and_points[ix]
            if gt_times_and_points is not None:
                ex_gt_times_and_points = gt_times_and_points[ix]
            if ex_pred_points is None and gt_pred_points is None and ex_pred_bboxes is None and ex_gt_bboxes is None and ex_pred_times_and_points is None and ex_gt_times_and_points is None:
                row["image"] = f"<img style=\"max-height:500px;max-width:500px;height:auto;width:auto;\" src={image_src}><img>"
            else:
                to_show = []
                if ex_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(ex_pred_points), "blue", format="xyxy"))
                if gt_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(gt_pred_points, 3), "green", format="xyxy"))
                if ex_pred_bboxes is not None:
                    to_show.append(BoxesToVisualize(ex_pred_bboxes, "blue", format="xyxy"))
                if ex_gt_bboxes is not None:
                    to_show.append(BoxesToVisualize(ex_gt_bboxes, "green", format="xyxy"))
                if ex_pred_times_and_points:
                    video_w, video_h = metadata['image_size']
                    coords = get_image_collage_coords_from_video_points(
                        ex_pred_times_and_points,
                        video_w, video_h,
                        fps=metadata.get("fake_timestamp_fps", 2),
                    )
                    to_show.extend([BoxesToVisualize(
                        [[abs_x-4, abs_y-4, abs_x+4, abs_y+4]], "blue", format="xyxy"
                    ) for abs_x, abs_y in coords])
                if ex_gt_times_and_points:
                    video_w, video_h = metadata['image_size']
                    coords = get_image_collage_coords_from_video_points(
                        ex_gt_times_and_points,
                        video_w, video_h,
                        fps=metadata.get("fake_timestamp_fps", 2),
                    )
                    to_show.extend([BoxesToVisualize(
                        [[abs_x-5, abs_y-5, abs_x+5, abs_y+5]], "green", format="xyxy"
                    ) for abs_x, abs_y in coords])
                row["image"] = get_html_image_with_boxes(image_src, to_show)
        row["prompt"] = html_escape(prompt_text)
        row["prediction"] = html_escape(pred_txt)

        if bmm is not None:
            row['bmm'] = html_escape(", ".join([f"{bmm_score:0.3f}" for bmm_score in bmm[ix]]))
        if high_res_indices is not None:
            row['high_res_indices'] = html_escape(", ".join([str(int(score)) for score in high_res_indices[ix]]))

        if "answers" in metadata:
            gt = metadata["answers"]
        elif "answer" in metadata:
            gt = metadata["answer"]
        elif "caption" in metadata:
            gt = metadata["caption"]
        else:
            gt = None
        if gt is not None:
            if isinstance(gt, list):
                gt = "<br>".join(html_escape(x) for x in gt)
            else:
                gt = html_escape(gt)
            row["gt"] = gt
        if scores is not None:
            if isinstance(scores[ix], dict):
                for k, v in scores[ix].items():
                    if isinstance(v, str):
                        row[k] = v
                    elif isinstance(v, list):
                        row[k] = "<br>".join(html_escape(x) for x in v)
                    else:
                        row[k] = "" if v is None else f"{v:0.3f}"
            else:
                row["score"] = f"{scores[ix]:0.3f}"

        if "display_in_eval" in metadata and metadata["display_in_eval"]:
            copied_metadata = copy.deepcopy(metadata)
            if "image" in copied_metadata:
                copied_metadata.pop("image")
            row['input_metadata'] = json.dumps(copied_metadata)

        rows.append(row)
    return HtmlTable(rows)


def get_gcs_url(output_file):
    assert output_file.startswith("gs://")
    return f"https://storage.cloud.google.com/{output_file[5:]}?authuser=1"


class Evaluator:
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        raise NotImplementedError()


class SavePredictions(Evaluator):

    @staticmethod
    def get_file_name(step, process_index):
        filename = ""
        if step is not None:
            filename += f"step{step}-"
        if get_world_size() > 1 and process_index is not None:
            filename += f"shard{process_index}"
        filename += "predictions"
        return filename

    def __init__(self, output_dir, json=True, save_tokens=True,
                 log_examples=10, table=100):
        self.save_tokens = save_tokens
        self.output_dir = output_dir
        self.log_examples = log_examples
        self.json = json
        self.table = table

    def __call__(self, metadatas, predictions, tokenizer,
                 step=None, scores=None):
        if not self.output_dir.startswith("gs://"):
            if not os.path.exists(self.output_dir):
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        json_data = []
        html_data = []

        n_no_eos = 0
        for tok in new_tokens:
            if not np.any(tok == tokenizer.eos_token_id):
                n_no_eos += 1
        if n_no_eos > 0:
            logging.warning(f"{n_no_eos}/{len(new_tokens)} ({n_no_eos/len(new_tokens):00.4f}) "
                            f"examples have no EOS, your inference tokens might be too short")

        for ex_ix, pred_seq in enumerate(new_tokens):
            text = tokenizer.decode(pred_seq[pred_seq >= 0])
            json_row = dict(prediction=text)
            if self.save_tokens:
                json_row["n_tokens"] = pred_seq.tolist()
            prompt_text = postprocess_prompt(tokenizer.decode(prompt_tokens[ex_ix][prompt_tokens[ex_ix] >= 0]))
            if tokenizer.adds_space:
                sep = " "
            else:
                sep = ""
            json_row["prompt"] = prompt_text
            if "bmm" in predictions:
                json_row["bmm"] = predictions["bmm"][ex_ix].tolist()
            if "high_res_indices" in predictions:
                json_row["high_res_indices"] = predictions["high_res_indices"][ex_ix].tolist()

            metadata = metadatas[ex_ix]
            if ex_ix < self.log_examples:
                log.info("*"*30)
                if "example_id" in metadata:
                    log.info(metadata['example_id'])
                log.info(' '.join((prompt_text + sep + text.replace("\n", "\\n")).split()))
            json_row.update({k: v for k, v in metadata.items() if isinstance(v, (str, float, int))})
            json_data.append(json_row)

        json_file = None
        html_file = None
        metrics = {}

        if self.json:
            log.info("Save prediction JSON")
            if get_world_size() > 1 and self.json:
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(json_data, all_predictions)
                    json_data = flatten_list(all_predictions)
                else:
                    dist.gather_object(json_data, None)
            if get_global_rank() == 0:
                write_file(
                    self.output_dir,
                    self.get_file_name(step, None) + ".json",
                    json.dumps(json_data, indent=2),
                    save_overwrite=True
                )
                log.info("done saving json")

        if self.table:
            metrics["prediction_table"] = gather_examples_as_html(self.table, tokenizer, metadatas, predictions)
        return metrics


def is_point_in_region(point: Tuple[float, float], mask: np.ndarray) -> bool:
    """
    Check if the point (x, y) is within the region defined by the boolean mask.

    Parameters:
    - point (tuple of floats): x/y-coordinate of the point
    - mask (2D numpy array): Boolean mask of shape [H, W] representing the region

    Returns:
    - bool: True if the point is within the region, False otherwise
    """
    height, width = mask.shape
    x, y = point

    # Round the coordinates to the nearest integer
    x_int = int(round(x))
    y_int = int(round(y))

    # Check if the rounded point is within the bounds of the image
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False

    # Check if the point is within the region
    return mask[y_int, x_int]


def compute_precision(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(preds)


def compute_recall(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(masks)


def f1_score(precision: float, recall: float, epsilon: float = 1e-10):
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + epsilon)


# Metric function aiming to replicate the step-wise accuracy as described in Appendix D.3 of
# the AndroidControl paper. Very specific to the action space and edge cases in that dataset.
def compute_stepwise_accuracy(ground_truth, predictions, target_bbs):
    def parse_action(action_str):
        action_str = action_str.strip().lower()

        def get_coords(action):
            coords = re.findall(r'\d+(?:\.\d+)?', action)
            return (float(coords[0]), float(coords[1])) if len(coords) >= 2 else ''

        if action_str.startswith("click"):
            return {"type": "click", "coords": get_coords(action_str)}
        elif action_str.startswith("long press"):
            return {"type": "long press", "coords": get_coords(action_str)}
        elif action_str.startswith("type"):
            return {"type": "type", "text": action_str[5:]}
        elif action_str.startswith("scroll"):
            return {"type": "scroll", "direction": action_str.split()[1] if len(action_str.split()) >= 2 else ''}
        elif action_str == "wait":
            return {"type": "wait"}
        elif action_str.startswith("open app"):
            return {"type": "open_app", "app_name": action_str[9:]}
        elif action_str == "navigate home":
            return {"type": "navigate home"}
        elif action_str == "navigate back":
            return {"type": "navigate back"}
        return {"type": None}

    def within_bounding_box(coords, box):
        x, y = coords[0], coords[1]
        bbox_values = re.findall(r'\d+\.\d+', box)
        x1, y1, x2, y2 = [float(val) for val in bbox_values]
        return x1 <= x <= x2 and y1 <= y <= y2

    all_predictions = []
    metrics = []
    for gt_action, pred_action, gt_box in zip(ground_truth, predictions, target_bbs):
        metric = 'incorrect'  # default to a prediction being incorrect until proven otherwise

        gt_parsed = parse_action(gt_action)
        pred_parsed = parse_action(pred_action)
        correct_predictions = 0
        if gt_parsed["type"] == pred_parsed["type"]:
            if gt_parsed["type"] in ["click", "long press"]:
                gt_coords = gt_parsed["coords"]
                if "coords" in pred_parsed and pred_parsed['coords'] != '' and gt_box != None and gt_parsed['coords'] != '':
                    pred_coords = pred_parsed["coords"]
                    if within_bounding_box(pred_coords, gt_box):
                        correct_predictions += 1
                        metric = 'correct'
            elif gt_parsed["type"] == "type" and gt_parsed["text"] == pred_parsed["text"]:
                correct_predictions += 1
                metric = 'correct'
            elif gt_parsed["type"] == "scroll" and gt_parsed["direction"] == pred_parsed["direction"]:
                correct_predictions += 1
                metric = 'correct'
            elif gt_parsed["type"] in ["navigate home", "navigate back", "wait"]:
                correct_predictions += 1  # These actions have no parameters to compare
                metric = 'correct'
            elif gt_parsed["type"] == "open_app" and pred_parsed["app_name"] == gt_parsed["app_name"]:
                correct_predictions += 1
                metric = 'correct'
            else:
                if gt_parsed == pred_parsed:
                    correct_predictions += 1
                    metric = 'correct'
        else:
            # Consider open_app and click on app name equivalent
            if pred_parsed["type"] == "click" and gt_parsed["type"] == "open_app":
                if gt_box not in [None, ''] and "coords" in pred_parsed and pred_parsed['coords'] != '':
                    if within_bounding_box(pred_parsed['coords'], gt_box):
                        correct_predictions += 1
                        metric = 'correct'
        all_predictions.append(correct_predictions)
        metrics.append(metric)
    # max with 1 since its technically possible for a node to get 0 valid examples
    return all_predictions, metrics


class PointingEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        model_predictions = predictions["predictions_text"]
        scores = defaultdict(list)
        pred_points = []
        gt_points = []
        for ex_ix, pred in enumerate(model_predictions):
            metadata = metadatas[ex_ix]
            answer_points = metadata["gt_points"]
            masks = metadata["masks"]
            image_w, image_h = metadata["image_size"]
            abs_preds = extract_points(pred, image_w, image_h)

            if len(answer_points) == 0:
                precision = recall = f1 = float(abs_preds is None or len(abs_preds) == 0)
                abs_gts = None
            else:
                abs_gts = answer_points
                if len(abs_preds) == 0:
                    precision = recall = f1 = 0.0
                else:
                    abs_preds = np.array(abs_preds)
                    dists = cdist(abs_preds, abs_gts)
                    row_ind, col_ind = linear_sum_assignment(dists)
                    precision = compute_precision(row_ind, col_ind, abs_preds, masks)
                    recall = compute_recall(row_ind, col_ind, abs_preds, masks)
                    f1 = f1_score(precision, recall)
            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)

            pred_points.append(abs_preds)
            gt_points.append(abs_gts)

        out = {}

        if "was_lowered" in metadatas[0]:
            # Get a score with and without lowering
            lowered = np.array([(x["was_lowered"] or x["was_lowered"] is None) for x in metadatas])
            nocase = np.array([(not x["was_lowered"] or x["was_lowered"] is None) for x in metadatas])
            for k, v in scores.items():
                v = np.array(v)
                out[k] = mean_metric(v[nocase])
                out[f"{k}_lower"] = mean_metric(v[lowered])
        else:
            for k, v in scores.items():
                out[k] = mean_metric(v)

        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(model_predictions))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, per_example_scores,
                pred_points=pred_points, gt_points=gt_points
            )
        return out


class PointCountEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        model_outputs = predictions["predictions_text"]
        vocab = tokenizer
        all_scores = defaultdict(list)
        per_category_scores = defaultdict(list)
        gt_counts_per_device = []
        pred_points = []
        gt_points = []
        for ex_ix, original_pred in enumerate(model_outputs):
            metadata = metadatas[ex_ix]
            pred = original_pred.lower().rstrip(".").strip()
            gt = metadata["count"]

            pred_int = None
            parts = pred.split()

            if parts:
                try:
                    pred_int = int(parts[-1].strip(". "))
                except ValueError:
                    pass

                if pred_int is None:
                    if parts[-1] in WORD_TO_NUM:
                        pred_int = WORD_TO_NUM[parts[-1]]

            if pred_int is None:
                match = re.match(".*a total of ([0-9]+).*", pred)
                if match:
                    pred_int = int(match.group(1))

            if pred_int is None:
                match = re.match(".*\\bnone\\b.*", pred, re.IGNORECASE)
                if match:
                    pred_int = 0

            pred_int = len(extract_points(pred, 100, 100))

            if pred_int is None:
                correct, close, valid = 0, 0, False
            else:
                correct = gt == pred_int
                close = abs(gt - pred_int) <= 1
                valid = True
            all_scores["close"].append(close)
            all_scores["valid"].append(valid)
            all_scores["correct"].append(correct)

            per_category_scores[int(gt)].append(correct)
            gt_counts_per_device.append(int(gt))

            abs_preds = None
            abs_gts = None
            if "image_size" in metadata:
                image_w, image_h = metadata["image_size"]
                try:
                    if len(re.findall(r"(\d+\.\d+),\s*(\d+\.\d+)", original_pred)) > 0:
                        abs_preds = np.array(extract_points(original_pred, image_w, image_h))
                except Exception as e:
                    print("Failed extracting pred points with error - ", e)
                    abs_preds = None
                try:
                    if "points" in metadata:
                        abs_gts = metadata["metadata/points"]
                except Exception as e:
                    print("Failed extracting gt points with error - ", e)
                    abs_gts = None

            pred_points.append(abs_preds)
            gt_points.append(abs_gts)

        num_examples_per_device = torch.tensor(len(model_outputs), dtype=torch.int32, device=torch.device("cuda"))
        num_examples = torch.zeros(get_world_size(), dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(num_examples, num_examples_per_device)
        max_num_examples = num_examples.detach().cpu().max().item()
        gt_counts_per_device = torch.tensor(gt_counts_per_device, dtype=torch.int32, device=torch.device("cuda"))
        gt_counts_per_device = torch.cat(
            [gt_counts_per_device, torch.full((max_num_examples - len(model_outputs),), -1, dtype=torch.int32, device=torch.device("cuda"))],
            dim=0,
        )
        gt_counts = torch.zeros(get_world_size() * max_num_examples, dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(gt_counts, gt_counts_per_device)
        gt_counts = gt_counts.detach().cpu().numpy()
        gt_counts = np.sort(np.unique(gt_counts[gt_counts >= 0]))

        out = {}
        for k, v in all_scores.items():
            out[k] = mean_metric(v)

        for k in gt_counts:
            out[f"correct_{k}"] = mean_metric(per_category_scores[k])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, all_scores["correct"],
                pred_points=pred_points, gt_points=gt_points
            )
        return out


def _math_vista_score(args):
    return math_vista_score(*args)


class MathVistaEval(Evaluator):
    def __init__(self, n_to_log=None, n_threads=4):
        self.n_to_log = n_to_log
        self.n_threads = n_threads

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer

        _args = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            _args.append((pred, metadatas[ex_ix], get_openai_key()))

        scores = []
        barrier()
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            for score in pool.map(_math_vista_score, _args):
                scores.append(score)
        barrier()

        out = dict(score=mean_metric(scores))
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores)
        return out


class VqaEval(Evaluator):

    def __init__(self, score_fn=("vqa_score",), n_to_log=None):
        self.metric = score_fn
        assert len(set(self.metric)) == len(self.metric)
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred in enumerate(response_text):
            metadata = metadatas[ex_ix]
            if "answer" in metadata:
                answers = metadata["answer"]
            elif "answers" in metadata:
                answers = metadata["answers"]
            else:
                answers = None
            if isinstance(answers, str):
                answers = [answers]

            pred = pred.strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
                pred_long = pred
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())

            for metric in self.metric:
                if metric == "vqa_score":
                    score = vqa_score(answers, pred)
                elif metric == 'llm_judge':
                    score = llm_judge_score(answers, pred, get_openai_key())
                elif metric == "ansl":
                    score = max(anls_metric(ref, pred) for ref in answers)
                elif metric == "relaxed_correctness":
                    score = max(relaxed_correctness(ans, pred) for ans in answers)
                elif metric == "scifi_relaxed_correctness":
                    score = max(scifi_relaxed_correctness(ans, pred) for ans in answers)
                elif metric == "a_okvqa_score":
                    score = a_okvqa_score(answers, pred)
                elif metric == "em":
                    score = pred.lower() in [x.lower() for x in answers]
                elif metric == "em_start":
                    pred = pred.lower()
                    pred = pred.strip().lstrip()  # deal with " B. ped"

                    answer = answers[0].lower().strip().lstrip()  # match "B." to even "B)" or "B"
                    answer = answer[0]

                    # Limitation - might match even if pred is "A ball is seen" and GT is A.
                    score = pred.startswith(answer)

                elif metric == "mc":
                    options = metadata["option_names"]
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["answer_idx"]
                elif metric == "perception_test_mc":
                    get_answer_idx = select_perception_test_option(pred)
                    score = get_answer_idx == metadata["answer_idx"]
                elif metric == "ego_schema_mc":
                    options = metadata["options"]
                    get_answer_idx = select_ego_schema_option(pred, options)
                    score = int(get_answer_idx) == int(metadata["answer_idx"])
                elif metric == "nextqa_mc":
                    options = metadata["options"]
                    answer = answers[0]
                    score = nextqa_mc(answer, pred, options)
                elif metric == "muir_bench_mc":
                    options = metadata["options"]
                    answer = string.ascii_uppercase[metadata["answer_idx"]]
                    score = muir_bench_mc(answer, pred, options)
                elif metric == "mantis_eval_mc":
                    options = metadata["options"]
                    answer = answers[0]
                    question_type = metadata["question_type"]
                    score = mantis_eval_mc(answer, pred, options, question_type)
                elif metric in ["mc_ai2d_transparent", "mc_ai2d_opaque"]: # mc split by transparency
                    has_transparent_box = metadata["has_transparent_box"]
                    abc_label = metadata["abc_label"]
                    # for abc_label, either evaluate on opaque or transparent boxes
                    if abc_label:
                        if metric == "mc_ai2d_transparent" and not has_transparent_box:
                            continue
                        elif metric == "mc_ai2d_opaque" and has_transparent_box:
                            continue
                    options = metadata["option_names"]
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["answer_idx"]
                elif metric == "mmmu_score":
                    score = mmmu_score(answers, pred, metadata)
                elif metric == "real_world_qa_score":
                    score = real_world_qa_score(metadata["answer"], pred, metadata)
                elif metric == "seed_bench_score":
                    options = list("abcd".upper())
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["metadata/answer_idx"][ex_ix]
                    data_type = metadata["metadata/data_type"][ex_ix].decode("utf-8")
                    score_lists[f"seed_bench_{data_type}_score"].append(score)
                elif metric == "math_vista_score":
                    score = math_vista_score(metadata["answer"], pred, metadata, get_openai_key())
                else:
                    raise NotImplementedError(metric)
                score_lists[metric].append(score)
                if "sample_difficulty" in metadata:
                    score_lists[f"{metric}_{metadata['sample_difficulty']}"].append(score)
                # FIXME Removed for now since this breaks multi-node eval by leading to different
                #  GPUs having different metrics
                # if "task_type" in metadata:
                #     score_lists[f"{metric}_{metadata['task_type']}"].append(score)

        if "is_human" in metadatas[0]:
            is_human = np.array([x["is_human"] for x in metadatas])
            for k, v in list(score_lists.items()):
                score_lists[f"{k}_human"] = np.array(v)[is_human]
                score_lists[f"{k}_aug"] = np.array(v)[np.logical_not(is_human)]

        out = {}
        for k, v in score_lists.items():
            out[k] = mean_metric(v)

        if self.n_to_log:
            score_to_log = score_lists[self.metric[0]]
            if len(score_to_log) != len(metadatas):
                score_to_log = None
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_to_log)
        return out


WORD_TO_NUM = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'zero': 0,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20
}


class CountEval:

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadata, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        all_scores = defaultdict(list)
        points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadata[ex_ix]
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            pred = pred.split()[0].rstrip(".,")
            gt = metadata["count"]
            if "image_size" in metadata:
                w, h = metadata["image_size"]
                points.append(extract_points(pred, w, h))

            pred_int = None
            try:
                pred_int = int(pred)
            except ValueError:
                pass
            pred_int = 1

            if pred_int is None:
                pred = pred.lower()
                if pred in WORD_TO_NUM:
                    pred_int = WORD_TO_NUM[pred]

            # Parse out the int for point and count data
            if pred_int is None:
                match = re.match(".*a total of ([0-9]+).*", pred)
                if match:
                    pred_int = int(match.group(1))

            if pred_int is None:
                match = re.match(".*\\bnone\\b.*", pred, re.IGNORECASE)
                if match:
                    pred_int = 0

            if pred_int is None:
                correct, close, valid = 0, 0, False
            else:
                correct = gt == pred_int
                close = abs(gt - pred_int) <= 1
                valid = True
            all_scores["close"].append(close)
            all_scores["valid"].append(valid)
            all_scores["correct"].append(correct)

        out = {}
        for k, vals in all_scores.items():
            out[k] = mean_metric(vals)
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadata, predictions, all_scores["correct"],
                all_scores["valid"], pred_points=points
            )
        return out


class ClockEval:
    METRICS = [
        "overall_close",
        "overall_exact",
        "correctly_declines_to_answer",
        "all_correct",
        "all_close",
        "hour_correct",
        "minute_correct",
        "second_correct",
        "minute_close",
        "second_close",
    ]

    def __init__(self, n_to_log=None, is_test=False):
        self.n_to_log = n_to_log
        self.is_test = is_test

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        err_threshold = 1 if self.is_test else 3
        all_scores = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            metadata = metadatas[ex_ix]
            scores = {}
            hour = metadata["hour"]
            if self.is_test and hour == 12:
                hour = 0
            minute = metadata["minute"]
            if self.is_test and minute == 60:
                minute = 0
            second = metadata["second"]
            answerable = hour > -1 or minute > -1 or second > -1

            scores["gt"] = f"{hour}:{minute}:{second}"
            if not answerable:
                scores["correctly_declines_to_answer"] = ":" not in pred
                scores["overall_close"] = scores["correctly_declines_to_answer"]
                scores["overall_exact"] = scores["correctly_declines_to_answer"]
                all_scores.append(scores)
                continue

            # pred = inputs["metadata/text"][ex_ix].decode("utf-8")
            parts = pred.split(":")
            try:
                pred_hour = int(parts[0].split()[-1])
                if "PM" in pred and not self.is_test:
                    pred_hour += 12
                if self.is_test and pred_hour >= 12:
                    pred_hour -= 12
                hour_correct = (pred_hour == hour) or (hour == 0 and pred_hour == 12)
            except (ValueError, IndexError):
                hour_correct = False
            scores["hour_correct"] = hour_correct

            try:
                minute_pred = int(parts[1].split()[0])
                if self.is_test and minute_pred == 60:
                    minute_pred = 0
                minute_correct = minute_pred == minute
                minute_close = abs(minute_pred - minute) <= err_threshold
                if self.is_test:
                    minute_close = minute_close or abs(minute_pred - minute) == 59
            except (ValueError, IndexError):
                minute_correct = 0
                minute_close = 0
            scores["minute_correct"] = minute_correct
            scores["minute_close"] = minute_close

            if second != -1:
                try:
                    second_pred = int(parts[2].split()[0])
                    second_correct = second_pred == second
                    second_close = abs(second_pred - second) <= err_threshold
                except (ValueError, IndexError):
                    second_correct = 0
                    second_close = 0
                scores["second_correct"] = second_correct
                scores["second_close"] = second_close
                scores["all_correct"] = minute_correct and hour_correct and second_correct
                scores["all_close"] = minute_close and hour_correct and second_close
            else:
                scores["all_correct"] = minute_correct and hour_correct
                scores["all_close"] = minute_close and hour_correct
                if self.is_test:
                    scores["all_close"] = scores["all_close"] or (
                        abs(hour * 60 + minute - (pred_hour * 60 + minute_pred)) == 719
                    )

            # FIXME can we remove?
            scores["overall_close"] = scores["all_close"]
            scores["overall_exact"] = scores["all_correct"]
            all_scores.append(scores)

        to_show = []
        for score in all_scores:
            to_show.append({k: score[k] for k in ["overall_close", "overall_exact"]})

        out = {}
        for k in self.METRICS:
            out[k] = mean_metric([x[k] for x in all_scores if k in x])
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, to_show)
        return out


def compute_area(bbox: list, invalid: float=None) -> float:
    x1, y1, x2, y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def compute_iou(bbox1: list, bbox2: list, verbose: bool=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1, invalid=0)
    area2 = compute_area(bbox2, invalid=0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou


TEMPORAL_ASPECTS = [
    "action",
    "direction",
    "speed",
    "order",
    "attribute_change",
]


FINE_GRAINED_TEMPORAL_ASPECTS = [
    "fine-grained action",
    "coarse-grained action",
    "object motion",
    "camera motion",
    "absolute speed",
    "relative speed",
    "order",
    "color & light change",
    "size & shape change",
    "combined change",
    "other change",
]

TEMP_COMPASS_TASKS = ["multi-choice", "yes_no", "caption_matching", "captioning"]


class QVHighlightsEval(Evaluator):
    def __init__(self, n_to_log=None, iou_threshold=0.5):
        self.n_to_log = n_to_log
        self.iou_threshold = iou_threshold

    def parse_segments(self, text):
        """
        Parse temporal segments from text output.
        Expected format: "[start1-end1], [start2-end2], ..."
        Returns a list of [start, end] pairs.
        """
        segments = []
        # Look for patterns like [X-Y] or (X-Y) or X-Y
        pattern = r'[\[\(]?(\d+\.?\d*)\s*-\s*(\d+\.?\d*)[\]\)]?'
        matches = re.finditer(pattern, text)

        for match in matches:
            try:
                start = float(match.group(1))
                end = float(match.group(2))

                # Limitation: If the model generates invalid segments, the evaluation does not penalise that
                if start < end:  # Ensure valid segment.
                    segments.append([start, end])
            except (ValueError, IndexError):
                continue

        return segments

    def compute_iou(self, segment1, segment2):
        """
        Compute IoU (Intersection over Union) between two temporal segments.
        """
        start1, end1 = segment1
        start2, end2 = segment2

        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        if intersection_end <= intersection_start:
            return 0.0  # No intersection

        intersection = intersection_end - intersection_start

        # Calculate union
        union = (end1 - start1) + (end2 - start2) - intersection

        return intersection / union if union > 0 else 0.0

    def compute_metrics(self, pred_segments, gt_segments):
        """
        Compute precision, recall, and F1 score for predicted segments.
        Uses IoU threshold to determine if a prediction matches a ground truth segment.
        """
        if not gt_segments:
            if not pred_segments:
                return 1.0, 1.0, 1.0  # Both empty, perfect match
            return 0.0, 0.0, 0.0  # No ground truth but predictions exist

        if not pred_segments:
            return 0.0, 0.0, 0.0  # No predictions

        # For each ground truth segment, find the best matching prediction
        gt_matched = [False] * len(gt_segments)
        pred_matched = [False] * len(pred_segments)

        # Create IoU matrix
        iou_matrix = np.zeros((len(gt_segments), len(pred_segments)))
        for i, gt_seg in enumerate(gt_segments):
            for j, pred_seg in enumerate(pred_segments):
                iou_matrix[i, j] = self.compute_iou(gt_seg, pred_seg)

        # Match segments greedily based on IoU
        for i in range(len(gt_segments)):
            for j in range(len(pred_segments)):
                if iou_matrix[i, j] >= self.iou_threshold and not gt_matched[i] and not pred_matched[j]:
                    gt_matched[i] = True
                    pred_matched[j] = True

        # Calculate metrics
        true_positives = sum(pred_matched)
        precision = true_positives / len(pred_segments) if pred_segments else 0.0
        recall = true_positives / len(gt_segments) if gt_segments else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        precisions = []
        recalls = []
        f1_scores = []
        referred_avg = []
        not_referred_avg = []
        saliency_diff_avg = []

        bmm = predictions["bmm"] if "bmm" in predictions else None
        # get the frame times
        frame_time_stamps = predictions["frame_time_stamps"] if "frame_time_stamps" in predictions else None

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = vocab.decode(pred_seq[pred_seq >= 0]).strip()

            # Get ground truth segments
            gt_segments = metadata["relevant_windows"]

            if bmm and frame_time_stamps:
                frame_time_stamps_instance = frame_time_stamps[ex_ix]
                num_frames = len(frame_time_stamps_instance)
                frame_time_mask = np.zeros(num_frames, dtype=bool)

                for gt_instance in gt_segments:
                    frame_time_mask[(gt_instance[0] <= frame_time_stamps_instance) & (frame_time_stamps_instance <= gt_instance[1])] = True

                predicted_frame_scores = bmm[ex_ix][:num_frames]
                referred_scores = predicted_frame_scores[frame_time_mask]
                not_referred_scores = predicted_frame_scores[~frame_time_mask]

                if len(referred_scores) > 0:  # Only add scores if there are referred regions
                    referred_avg.append(np.sum(referred_scores) / (len(referred_scores) + 1e-6))
                if len(not_referred_scores) > 0:  # Only add scores if there are non referred regions
                    not_referred_avg.append(np.sum(not_referred_scores) / (len(not_referred_scores) + 1e-6))

                num_gt_frames = len(metadata["scaled_avg_scores"])
                if num_gt_frames == num_frames:
                    saliency_score_diff = np.sum(np.abs(predicted_frame_scores - metadata["scaled_avg_scores"]))
                elif num_frames > num_gt_frames:  # Interpolate to the set with the higher count
                    saliency_score_diff = np.sum(np.abs(predicted_frame_scores - interpolate_frame_scores(metadata["scaled_avg_scores"], num_frames)))
                else:
                    saliency_score_diff = np.sum(np.abs(interpolate_frame_scores(predicted_frame_scores, num_gt_frames) - metadata["scaled_avg_scores"]))
                saliency_diff_avg.append(saliency_score_diff / num_frames)

            # Parse predicted segments
            pred_segments = self.parse_segments(pred_text)

            # Compute metrics
            precision, recall, f1 = self.compute_metrics(pred_segments, gt_segments)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Aggregate metrics
        out = {
            "precision": mean_metric(precisions),
            "recall": mean_metric(recalls),
            "f1": mean_metric(f1_scores)
        }
        if referred_avg and not_referred_avg:
            out["referred_avg"] = mean_metric(referred_avg)
            out["not_referred_avg"] = mean_metric(not_referred_avg)
        if len(saliency_diff_avg) > 0:
            out["saliency_diff_avg"] = mean_metric(saliency_diff_avg)

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, f1_scores
            )

        return out


class TempCompassEval(Evaluator):
    TEMPORAL_ASPECTS = ['action', 'attribute_change', 'direction', 'order', 'speed']

    def __init__(self, task="all", disable_api=False, n_to_log=None):
        if task == "all":
            self.tasks = TEMP_COMPASS_TASKS
        elif task == "internal":
            self.tasks = ["multi-choice", "yes_no", "caption_matching"]
        else:
            self.tasks = [task]
        self.disable_api = disable_api
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(response_text):
            metadata = metadatas[ex_ix]
            task = metadata["task"]
            aspect = metadata['temporal_aspect']
            pred = pred_seq.strip()
            score = temp_compass_score(
                pred, metadata, get_openai_key(), use_api=not self.disable_api,
            )
            score_lists[task].append(score)
            score_lists["all"].append(score)
            score_lists[f"{task}_{aspect}"].append(score)

            if "sample_difficulty" in metadata:
                score_lists[f"{task}_{metadata['sample_difficulty']}"].append(score)
                score_lists[f"all_{metadata['sample_difficulty']}"].append(score)

        out = {}
        for k in self.tasks:
            out[k] = mean_metric(score_lists[k])
            if "sample_difficulty" in metadatas[0]:
                out[f"{k}_easy"] = mean_metric(score_lists[f"{k}_easy"])
                out[f"{k}_medium"] = mean_metric(score_lists[f"{k}_medium"])
                out[f"{k}_hard"] = mean_metric(score_lists[f"{k}_hard"])
            for aspect in self.TEMPORAL_ASPECTS:
                out[f"{k}_{aspect}"] = mean_metric(score_lists[f"{k}_{aspect}"])

        out["all"] = mean_metric(score_lists["all"])
        if "sample_difficulty" in metadatas[0]:
            out["all_easy"] = mean_metric(score_lists["all_easy"])
            out["all_medium"] = mean_metric(score_lists["all_medium"])
            out["all_hard"] = mean_metric(score_lists["all_hard"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_lists["all"]
            )
        return out


class TemporalBenchEval:
    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        vocab = tokenizer

        video_kind = []
        video_idx = []
        video_score = []
        for ex_ix, pred_seq in enumerate(response_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            if "option_names" in metadata:
                options = metadata["option_names"]
                get_answer_idx = select_mc_option(pred, options)
                score = get_answer_idx == metadata["answer_idx"]
            else:
                score = metadata["answer"] == metadata["answer"]
            video_kind.append(metadata["type"])
            video_idx.append(metadata["video"])
            video_score.append(score)

        out = dict(
            kind=video_kind,
            idx=video_idx,
            score=video_score
        )
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, video_score)
        return out


class TomatoEval:
    reasoning_type_choices = [
        "count",
        "direction",
        "rotation",
        "shape&trend",
        "velocity&frequency",
        "visual_cues"
    ]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer

        scores = {k: [] for k in self.reasoning_type_choices}
        all_scores = []
        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            options = metadata["option_names"]
            # Tomato uses a LLM to find the answer option, but our model is trained for short
            # answer formats should it should not be needed
            get_answer_idx = select_mc_option(pred, options)
            score = get_answer_idx == metadata["answer_idx"]
            all_scores.append(score)
            scores[metadatas[ex_ix]["reasoning_type"]].append(score)

        out = {k: mean_metric(v) for k, v in scores.items()}
        out["all"] = mean_metric(all_scores)
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, all_scores)
        return out


class VideoHallucerEval(Evaluator):
    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer
        scores = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()

            pred = pred.lower()
            pred = pred.strip().lstrip()

            # Extract yes/no answer from prediction
            if "yes" in pred.lower():
                pred_answer = "yes"
            elif "no" in pred.lower():
                pred_answer = "no"
            else:
                # If neither yes nor no is found, consider it incorrect
                pred_answer = ""

            answer = metadata["answer"].lower().strip().lstrip()

            # Check if prediction matches the expected answer
            score = pred_answer == answer

            # Track scores by question_group_id
            scores[metadata['question_group_id']].append(score)

        # Process scores by question_group_id
        question_to_score_tuples = [score_tuple for score_tuple in scores.items()]
        output_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(output_list, question_to_score_tuples)

        collected_scores = defaultdict(list)
        for output in output_list:
            for key, value in output:
                collected_scores[key].extend(value)

        m_b_acc = defaultdict(list)
        for key, item in collected_scores.items():
            m_b_acc['all'].append(0 if 0 in item else 1)

            type = key.split("-")[0]
            m_b_acc[type].append(0 if 0 in item else 1)

        # Calculate metrics
        out = {}
        for key, item in m_b_acc.items():
            out[f"m_b_acc_{key}"] = mean_metric(item)

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions
            )
        return out


VIDEO_MME_CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]


VIDEO_MME_SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]


VIDEO_MME_TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


class VideoMMEEval(Evaluator):
    TASK_TYPES = [
        'Action Reasoning', 'Action Recognition', 'Attribute Perception', 'Counting Problem',
        'Information Synopsis', 'OCR Problems', 'Object Reasoning', 'Object Recognition',
        'Spatial Perception', 'Spatial Reasoning', 'Temporal Perception', 'Temporal Reasoning'
    ]
    CATEGORIES = [
        'Acrobatics', 'Animation', 'Astronomy', 'Athletics', 'Basketball', 'Biology & Medicine',
        'Daily Life', 'Documentary', 'Esports', 'Exercise', 'Fashion', 'Finance & Commerce',
        'Food', 'Football', 'Geography', 'Handicraft', 'Humanity & History', 'Law', 'Life Tip',
        'Literature & Art', 'Magic Show', 'Movie & TV Show', 'Multilingual', 'News Report',
        'Other Sports', 'Pet & Animal', 'Stage Play', 'Technology', 'Travel', 'Variety Show'
    ]

    def __init__(self, duration="all", n_to_log=None):
        self.durations = ["short", "medium", "long"] if duration == "all" else [duration]
        self.n_to_log = n_to_log

    def extract_characters_regex(self, s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is"
            "The correct option is",
            "Best answer:"
            "Best option:",
            "Answer:",
            "Option:",
            "The correct answer",
            "The correct option",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCD]", s):
            return ""
        matches = re.search(r'[ABCD]', s)
        if matches is None:
            return ""
        return matches[0]

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            pred = self.extract_characters_regex(pred)
            answer = metadata["answer"]
            if pred == "":
                # I am not sure why, but the original code skipped if the extraction is an empty string
                continue
            score = pred == answer

            score_lists["all"].append(score)

            duration = metadata["duration"]
            score_lists[f"{duration}"].append(score)

            if "sample_difficulty" in metadata:
                score_lists[f"{duration}_{metadata['sample_difficulty']}"].append(score)
                score_lists[f"all_{metadata['sample_difficulty']}"].append(score)

            score_lists[f"all_category_{metadata['sub_category']}"].append(score)
            score_lists[f"all_task_{metadata['task_type']}"].append(score)

            if duration in ["medium", "long"]:
                score_lists["medium_long"].append(score)
                if "sample_difficulty" in metadata:
                    score_lists[f"medium_long_{metadata['sample_difficulty']}"].append(score)

        out = {}
        for k in self.durations:
            out[f"{k}"] = mean_metric(score_lists[f"{k}"])
        out["all"] = mean_metric(score_lists["all"])
        out["medium_long"] = mean_metric(score_lists["medium_long"])

        if "sample_difficulty" in metadatas[0]:
            for k in self.durations + ["all", "medium_long"]:
                out[f"{k}_easy"] = mean_metric(score_lists[f"{k}_easy"])
                out[f"{k}_medium"] = mean_metric(score_lists[f"{k}_medium"])
                out[f"{k}_hard"] = mean_metric(score_lists[f"{k}_hard"])

        for category in self.CATEGORIES:
            out[f"all_category_{category}"] = mean_metric(score_lists[f"all_category_{category}"])
        for task_type in self.TASK_TYPES:
            out[f"all_task_{task_type}"] = mean_metric(score_lists[f"all_task_{task_type}"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_lists["all"]
            )
        return out


class MMEVideoOCREval(Evaluator):
    question_categories = [
        'Text_Recognition', 'Visual_Text_QA', 'Text_Grounding', 'Attribute_Recognition', 'Text_Based_Reasoning',
        'Change_Detection_and_Tracking', 'Special_Text_Parising', 'Robust_Video_Testing', 'Cross_Frame_Text_Understanding',
        'Text_Based_Video_Understanding'
    ]

    GPT_PROMPT = """You are a professional bilingual translation evaluator.

    Here are two sentences: one in Chinese and one in English.
    Sentence 1: {SENTENCE_1}
    Sentence 2: {SENTENCE_2}

    Please evaluate whether the two sentences convey the same meaning and can be considered accurate translations of each other.

    If the meanings are equivalent and the translation is accurate, respond with "correct".
    If there are significant differences in meaning or inaccuracies in translation, respond with "wrong".

    You must only respond with one word: "correct" or "wrong". Do not provide any explanations, comments, or additional text.
    Focus solely on semantic equivalence, not grammar or style. Ignore minor differences as long as the meaning is preserved."""

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
        from openai import OpenAI
        self.client = OpenAI(api_key=get_openai_key())

    def extract_characters_regex(self, s):
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is"
            "The correct option is",
            "Best answer:"
            "Best option:",
            "Answer:",
            "Option:",
            "The correct answer",
            "The correct option",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCD]", s):
            return ""
        matches = re.search(r'[ABCD]', s)
        if matches is None:
            return ""
        return matches[0]

    def get_chat_response(
            self,
            prompt: str,
            sys_prompt: str = "You are a helpful assistant.",
            max_tokens: int = 1024,
            temperature: float = 0.0,
            retries: int = 10,
):
        MODEL_VERSION = "gpt-4o-2024-08-06"
        client = self.client

        messages = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": MODEL_VERSION,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(**payload)
                content = response.choices[0].message.content.strip()
                return content
            except Exception as e:
                log.warning(f"Request failed: {e}")
                if attempt == retries - 1:
                    return ""

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            pred = self.extract_characters_regex(pred)
            ground_truth = metadata["answer"]
            if pred == "":
                # I am not sure why, but the original code skipped if the extraction is an empty string
                continue
            if metadata['eval_method'] == 'multiple_choice':
                score = pred == ground_truth
            elif metadata['eval_method'] == 'containment_match':
                if metadata['task'] == "trajectory_recognition" or metadata['task'] == "scrambled_recognition":
                    score = pred == ground_truth
                else:
                    ground_truth = ground_truth.replace("’", "'").lower()
                    pred = pred.replace("’", "'").lower()
                    if ";" in ground_truth:
                        answer_list = ground_truth.split(";")
                        answer_list = [ans.strip() for ans in answer_list]
                        answer_list = [ans.replace("’", "'") for ans in answer_list]
                        for ans in answer_list:
                            if ans not in pred:
                                print(f"ans: {ans} not in pred: {pred}")
                                score = 0.0
                                break
                        else:
                            score = 1.0
                    else:
                        if ground_truth in pred:
                            score = 1.0
                        else:
                            score = 0.0
            elif metadata['eval_method'] == "gpt_assisted_scoring":
                gpt_prompt = self.GPT_PROMPT.format(SENTENCE_1=ground_truth, SENTENCE_2=pred)
                score = -1
                try_num = 0
                while score == -1 and try_num <= 10:
                    try:
                        response = self.get_chat_response(prompt=gpt_prompt)
                        if "correct" in response.lower():
                            score = 1.0
                        elif "wrong" in response.lower():
                            score = 0.0
                        else:
                            score = -1
                            try_num += 1
                    except Exception as e:
                        log.warning(f"Error: {e}")
                        log.warning("Retrying...\n")
                if score == -1:
                    log.warning(f"GPT Error")
                    score = 0.0
            else:
                raise NotImplementedError
            score_lists["all"].append(score)
            question_category = metadata["question_category"]
            score_lists[question_category].append(score)


        out = {}
        for k in self.question_categories:
            out[f"{k}"] = mean_metric(score_lists[f"{k}"])
        out["all"] = mean_metric(score_lists["all"])


        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_lists["all"]
            )
        return out


def _mlvu_gen_score(args):
    task_type = args.pop("task_type")
    if task_type == "sub_scene":
        return mlvu_ssc_score(**args)
    else:
        return mlvu_summary_score(**args)


class MLVUGenEval(Evaluator):

    def __init__(self, n_to_log=None, n_threads=4):
        self.n_to_log = n_to_log
        self.n_threads = n_threads

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer

        _args = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            _args.append(
                dict(
                    task_type=metadatas[ex_ix]["task_type"],
                    prediction=pred,
                    metadata=metadatas[ex_ix],
                    openai_api_key=get_openai_key()
                )
            )

        scores = []
        barrier()
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            for score in pool.map(_mlvu_gen_score, _args):
                scores.append(score)
        barrier()

        score_lists = defaultdict(list)
        for score in scores:
            for k, v in score.items():
                score_lists[k].append(v)
        out = {k: mean_metric(v) for k, v in score_lists.items()}
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, [sum(score.values()) for score in scores]
            )
        return out


class LVBenchEval(Evaluator):
    TASK_TYPES = ['reasoning', 'temporal grounding', 'event understanding',
                  'summarization', 'key information retrieval',
                  'entity recognition']

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def eval_multi_choice(self, gold_i, pred_i):
        return gold_i.lower() == pred_i.lower()[:len(gold_i)] or gold_i.lower() == pred_i.lower()[-len(gold_i):]

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()

            pred = pred.lower()
            pred = pred.strip().lstrip()  # deal with " B. ped"

            answer = metadata["answer"].lower().strip().lstrip()  # match "B." to even "B)" or "B"
            answer = answer[0]

            # Limitation - might match even if pred is "A ball is seen" and GT is A.
            score = pred.startswith(answer)

            for qtype in metadata["qtype"]:
                score_lists[qtype].append(score)
            score_lists["all"].append(score)

        out = {}
        for task_type in self.TASK_TYPES:
            out[f"all_{task_type}"] = mean_metric(score_lists[f"{task_type}"])
        out["all"] = mean_metric(score_lists["all"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, score_lists["all"]
            )
        return out


class LongVideoBenchEval(Evaluator):
    DURATIONS = [15, 60, 600, 3600]
    TASK_TYPES = [
        'E2O', 'E3E', 'O2E', 'O3O', 'S2A', 'S2E', 'S2O', 'SAA', 'SOS', 'SSS',
        'T2A', 'T2E', 'T2O', 'T3E', 'T3O', 'TAA', 'TOS'
    ]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def parse_multi_choice_response(self, response, all_choices, index2ans):
        """
        Changed from MMMU-style complex parsing into simple parsing.
        Fixed to avoid 'D. A book' be parsed as A.
        Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
        """
        s = response.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:",
            "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCDE]", s):
            return random.choice(all_choices)

        matches = re.search(r"[ABCDE]", s)
        if matches is None:
            return random.choice(all_choices)
        return matches[0]

    def eval_multi_choice(self, gold_i, pred_i):
        correct = False
        # only they are exactly the same, we consider it as correct
        if isinstance(gold_i, list):
            for answer in gold_i:
                if answer == pred_i:
                    correct = True
                    break
        else:  # gold_i is a string
            if gold_i == pred_i:
                correct = True
        return correct

    def get_multi_choice_info(self, options):
        """
        Given the list of options for multiple choice question
        Return the index2ans and all_choices
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
        """

        start_chr = "A"
        all_choices = []
        index2ans = {}
        for i, option in enumerate(options):
            index2ans[chr(ord(start_chr) + i)] = option
            all_choices.append(chr(ord(start_chr) + i))

        return index2ans, all_choices

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            index2ans, all_choices = self.get_multi_choice_info(metadata["options"])

            parsed_pred = self.parse_multi_choice_response(pred, all_choices, index2ans)
            score = self.eval_multi_choice(metadata["answer"], parsed_pred)

            duration_group = metadata["duration_group"]
            task_type = metadata["question_category"]
            score_lists[f"duration_{duration_group}"].append(score)
            score_lists["all"].append(score)
            score_lists[f"all_{task_type}"].append(score)
            if "sample_difficulty" in metadata:
                score_lists[f"duration_{duration_group}_{metadata['sample_difficulty']}"].append(score)
                score_lists[f"all_{metadata['sample_difficulty']}"].append(score)

        out = {}
        for k in self.DURATIONS:
            out[f"duration_{k}"] = mean_metric(score_lists[f"duration_{k}"])
        out["all"] = mean_metric(score_lists["all"])
        for task_type in self.TASK_TYPES:
            out[f"all_{task_type}"] = mean_metric(score_lists[f"all_{task_type}"])
        if "sample_difficulty" in metadatas[0]:
            for k in self.DURATIONS:
                out[f"duration_{k}_easy"] = mean_metric(score_lists[f"duration_{k}_easy"])
                out[f"duration_{k}_medium"] = mean_metric(score_lists[f"duration_{k}_medium"])
                out[f"duration_{k}_hard"] = mean_metric(score_lists[f"duration_{k}_hard"])
        out["all_easy"] = mean_metric(score_lists["all_easy"])
        out["all_medium"] = mean_metric(score_lists["all_medium"])
        out["all_hard"] = mean_metric(score_lists["all_hard"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_lists["all"]
            )
        return out


class LongVideoBenchCaptionEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer

        scores_lists = {"recall":[], "consistency":[]}
        scores = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()

            scores_dict = eval_caption(pred, metadata, get_openai_key())
            scores_lists["recall"].append(scores_dict["recall"])
            scores_lists["consistency"].append(scores_dict["consistency"])
            scores.append(scores_dict)


        out = {}
        out["recall"] = mean_metric(scores_lists["recall"])
        out["consistency"] = mean_metric(scores_lists["consistency"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores
            )
        return out


class VinogroundEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def eval_multi_choice(self, gold_i, pred_i):
        return gold_i.lower() == pred_i.lower()[:len(gold_i)] or gold_i.lower() == pred_i.lower()[-len(gold_i):]

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            score = self.eval_multi_choice(metadata["answer"], pred)

            score_lists[metadata["qtype"]].append(score)
            score_lists[metadata["qtype"] + "_idx"].append(metadata["id"])

        out = {}
        out["text"] = score_lists["textscore"]
        out["video"] = score_lists["videoscore"]
        out["text_idx"] = score_lists["textscore_idx"]
        out["video_idx"] = score_lists["videoscore_idx"]

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, score_lists["textscore"]
            )
        return out


class VixMoCaptionEval(Evaluator):
    CATEGORIES = [
        "Action",
        "Attribute",
        "Camera",
        "Causation/Purpose",
        "Emotion/Affect",
        "Event",
        "Gesture",
        "Identity",
        "Lighting/Weather",
        "Location",
        "Motion/Trajectory",
        "OCR",
        "Object",
        "Pose",
        "Quantity/Number",
        "Relation",
        "Scene/Context",
        "State/Condition",
    ]

    def __init__(self, n_to_log=None, n_threads=32//get_local_world_size(), log_statements=False, version='v1'):
        self.n_to_log = n_to_log
        self.n_threads = n_threads
        self.log_statements = log_statements
        self.version = version

    def eval_v1(self, item_list):
        scores_lists = {"recall_w_gemini": [], "consistency_w_gemini": [], "recall_wo_gemini": [], "consistency_wo_gemini": []}
        category_recall_w_gemini = {c: [] for c in self.CATEGORIES}
        category_consistency_w_gemini = {c: [] for c in self.CATEGORIES}
        category_recall_wo_gemini = {c: [] for c in self.CATEGORIES}
        category_consistency_wo_gemini = {c: [] for c in self.CATEGORIES}
        scores = []

        barrier()
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(eval_vixmo_caption, item[0], item[1], item[2], item[3]): item
                for item in item_list
            }

            for future in tqdm(as_completed(futures),
                               total=len(futures)):
                scores_dict = future.result()
                if scores_dict is None:
                    continue

                scores_lists["recall_w_gemini"].append(scores_dict["recall_w_gemini"])
                scores_lists["consistency_w_gemini"].append(scores_dict["consistency_w_gemini"])
                scores_lists["recall_wo_gemini"].append(scores_dict["recall_wo_gemini"])
                scores_lists["consistency_wo_gemini"].append(scores_dict["consistency_wo_gemini"])

                for cate, r in scores_dict['category_recall_w_gemini'].items():
                    category_recall_w_gemini[cate].append(r)
                for cate, c in scores_dict['category_consistency_w_gemini'].items():
                    category_consistency_w_gemini[cate].append(c)
                for cate, r in scores_dict['category_recall_wo_gemini'].items():
                    category_recall_wo_gemini[cate].append(r)
                for cate, c in scores_dict['category_consistency_wo_gemini'].items():
                    category_consistency_wo_gemini[cate].append(c)

                score = {
                    "example_idx"          : scores_dict["example_idx"],
                    "recall_w_gemini"      : scores_dict["recall_w_gemini"],
                    "consistency_w_gemini" : scores_dict["consistency_w_gemini"],
                    "recall_wo_gemini"     : scores_dict["recall_wo_gemini"],
                    "consistency_wo_gemini": scores_dict["consistency_wo_gemini"],
                }
                if self.log_statements:
                    score["recall_statements_w_gemini"] = scores_dict["recall_statements_w_gemini"]
                    score["consistency_statements_w_gemini"] = scores_dict["consistency_statements_w_gemini"]
                    score["recall_statements_wo_gemini"] = scores_dict["recall_statements_wo_gemini"]
                    score["consistency_statements_wo_gemini"] = scores_dict["consistency_statements_wo_gemini"]
                scores.append(score)

        barrier()
        # reorder the scores to match the original order
        sorted_scores = sorted(scores, key=lambda x: x["example_idx"])

        out = {}
        out["recall_w_gemini"] = mean_metric(scores_lists["recall_w_gemini"])
        out["consistency_w_gemini"] = mean_metric(scores_lists["consistency_w_gemini"])
        out["recall_wo_gemini"] = mean_metric(scores_lists["recall_wo_gemini"])
        out["consistency_wo_gemini"] = mean_metric(scores_lists["consistency_wo_gemini"])

        for c in self.CATEGORIES:
            out[f"recall_w_gemini_{c}"] = mean_metric(category_recall_w_gemini[c])
            out[f"consistency_w_gemini_{c}"] = mean_metric(category_consistency_w_gemini[c])
            out[f"recall_wo_gemini_{c}"] = mean_metric(category_recall_wo_gemini[c])
            out[f"consistency_wo_gemini_{c}"] = mean_metric(category_consistency_wo_gemini[c])

        return out, sorted_scores

    def eval_v2(self, item_list):
        scores_lists = {"recall": [], "consistency": []}
        category_recall = {c: [] for c in self.CATEGORIES}
        category_consistency = {c: [] for c in self.CATEGORIES}
        scores = []

        barrier()
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(eval_vixmo_caption2, item[0], item[1], item[2], item[3]): item
                for item in item_list
            }

            for future in tqdm(as_completed(futures),
                               total=len(futures)):
                scores_dict = future.result()
                if scores_dict is None:
                    continue

                scores_lists["recall"].append(scores_dict["recall"])
                scores_lists["consistency"].append(scores_dict["consistency"])

                for cate, r in scores_dict['category_to_recall'].items():
                    category_recall[cate].append(r)
                for cate, c in scores_dict['category_to_consistency'].items():
                    category_consistency[cate].append(c)

                score = {
                    "example_idx"          : scores_dict["example_idx"],
                    "recall"      : scores_dict["recall"],
                    "consistency" : scores_dict["consistency"],
                }
                if self.log_statements:
                    score["recall_statements"] = scores_dict["recall_statements"]
                    score["consistency_statements"] = scores_dict["consistency_statements"]
                scores.append(score)

        barrier()
        # reorder the scores to match the original order
        sorted_scores = sorted(scores, key=lambda x: x["example_idx"])

        out = {}
        out["recall"] = mean_metric(scores_lists["recall"])
        out["consistency"] = mean_metric(scores_lists["consistency"])

        for c in self.CATEGORIES:
            out[f"recall_{c}"] = mean_metric(category_recall[c])
            out[f"consistency_{c}"] = mean_metric(category_consistency[c])

        return out, sorted_scores

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer
        item_list = []
        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            data_tuple = (ex_ix, pred, metadata, get_openai_key())
            item_list.append(data_tuple)

        if self.version == 'v1':
            out, sorted_scores = self.eval_v1(item_list)
        else:
            out, sorted_scores = self.eval_v2(item_list)

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, sorted_scores
            )

        return out

class VideoObjectTrackingEval(Evaluator):
    """
    Evaluator for video point tracking tasks using mask-based F1 validation.

    This evaluator processes model predictions for video point tracking and validates
    them against ground truth using segmentation masks. It uses the Hungarian algorithm
    for optimal point assignment and computes F1 scores based on mask overlap.

    Supported task name format:
        {dataset}_point_track_per_frame_fps_{fps}_sample_fps_{sampling_fps}

    Expected HF dataset path format:
        {VIDEO_DATA_HOME}/{dataset_name}_{point_type}_sample_fps_{sampling_fps}/

    Supported datasets: mevis, burst, ref-yt-vos, etc.
    """

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        # Segmentation dataset is already loaded during initialization

        scores = defaultdict(list)
        failed_videos = []

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            example_id = metadata['example_id']
            if "image_size" in metadata:
                width, height = metadata["image_size"]
            else:
                width, height = metadata["w"], metadata["h"]

            # Get ground truth
            # Ex: [{'frame': 0, 'time': '00:00.00', 'points': {0: {'point': [1038.0, 377.0], 'occluded': False}}}]
            gt_tracks = metadata['points']
            gt_masks = metadata['masks'] # Ex: {'mask_id': [mask_list per frame], ...}

            # Decode prediction to match GT format
            if isinstance(pred_seq, str):
                pred = pred_seq
            else:
                pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            video_fps = metadata['video_fps'] # Used for converting time to frame index
            pred_tracks = extract_tracks(
                pred, width, height, video_fps,
                format='video_point_track_per_frame'
            )

            # try:
            # Evaluate video point tracking by checking if it's in mask.
            video_metrics = evaluate_video_object_tracking(
                pred_tracks,
                gt_tracks,
                gt_masks,
                height,
                width
            )
            # Add to score lists
            scores["precision"].append(video_metrics['precision'])
            scores["recall"].append(video_metrics['recall'])
            scores["f1"].append(video_metrics['f1'])

            # hota metrics
            scores['DetA'].append(video_metrics['DetA'])
            scores['AssA'].append(video_metrics['AssA'])
            scores['HOTA'].append(video_metrics['HOTA'])

            # Get by category scores if available
            if 'category' in metadata:
                category = metadata['category']
                scores[f'precision_{category}'].append(video_metrics['precision'])
                scores[f'recall_{category}'].append(video_metrics['recall'])
                scores[f'f1_{category}'].append(video_metrics['f1'])
                scores[f'DetA_{category}'].append(video_metrics['DetA'])
                scores[f'AssA_{category}'].append(video_metrics['AssA'])
                scores[f'HOTA_{category}'].append(video_metrics['HOTA'])

            # except Exception as e:
            #     failed_videos.append((example_id, str(e)))
            #     log.warning(f"Error evaluating {example_id}: {e}")

        # Compute aggregated metrics
        out = {}
        for k, v in scores.items():
            out[k] = mean_metric(v)

        if failed_videos:
            log.warning(f"Failed to evaluate {len(failed_videos)} videos: {[x[0] for x in failed_videos[:5]]}")

        # Add visualization if requested
        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] if i < len(scores[k]) else 0.0
                                 for k in scores} for i in range(len(new_tokens))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores
            )

        return out


class Dream1KCaptionEval(Evaluator):

    def __init__(self, n_to_log=None, n_threads=32 // get_local_world_size(), log_statements=False):
        self.n_to_log = n_to_log
        self.n_threads = n_threads
        self.log_statements = log_statements

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        predictions_text = predictions["predictions_text"]
        vocab = tokenizer
        item_list = []
        for ex_ix, pred_seq in enumerate(predictions_text):
            metadata = metadatas[ex_ix]
            pred = pred_seq.strip()
            data_tuple = (ex_ix, pred, metadata, get_openai_key())
            item_list.append(data_tuple)

        scores_lists = {"recall": [], "consistency": []}
        scores = []

        barrier()
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(eval_dream_caption, item[0], item[1], item[2], item[3]): item
                for item in item_list
            }

            for future in tqdm(as_completed(futures),
                               total=len(futures)):
                scores_dict = future.result()
                if scores_dict is None:
                    continue

                scores_lists["recall"].append(scores_dict["score_r"])
                scores_lists["consistency"].append(scores_dict["score_p"])


                score = {
                    "example_idx"          : scores_dict["example_idx"],
                    "recall"      : scores_dict["score_r"],
                    "consistency" : scores_dict["score_p"],
                }
                scores.append(score)

        barrier()
        # reorder the scores to match the original order
        sorted_scores = sorted(scores, key=lambda x: x["example_idx"])

        out = {}
        out["recall"] = mean_metric(scores_lists["recall"])
        out["consistency"] = mean_metric(scores_lists["consistency"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, sorted_scores
            )

        return out


class MulSetEval(Evaluator):
    TASKS = {
        "mask": "Occlusion Restoration",
        "distance": "Distance Comparison",
        "direction": "Azimuth Transfer",
    }

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
    
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        scores = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            if "answer" in metadata:
                answers = metadata["answer"]
            elif "answers" in metadata:
                answers = metadata["answers"]
            else:
                answers = None
            if isinstance(answers, str):
                answers = [answers]

            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
                pred_long = pred
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())

            options = metadata["options"]
            answer = answers[0]
            score = muir_bench_mc(answer, pred, options)

            scores[self.TASKS[metadata["task"]]].append(score)
            scores["all"].append(score)
    
        out = {}
        for k in list(self.TASKS.values()):
            out[k] = mean_metric(scores[k])
        out["all"] = mean_metric(scores["all"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores["all"]
            )
        return out


class Ego3dBenchEval(Evaluator):
    NUMBER_CATEGORIES = [
        'Ego_Centric_Absolute_Distance',
        'Object_Centric_Absolute_Distance',
    ]
    MULTI_CHOICE_CATEGORIES = [
        'Ego_Centric_Absolute_Distance_MultiChoice',
        'Ego_Centric_Motion_Reasoning',
        'Ego_Centric_Relative_Distance',
        'Localization',
        'Object_Centric_Absolute_Distance_MultiChoice',
        'Object_Centric_Motion_Reasoning',
        'Object_Centric_Relative_Distance',
        'Travel_Time',
    ]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
    
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        scores = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            if "answer" in metadata:
                answers = metadata["answer"]
            elif "answers" in metadata:
                answers = metadata["answers"]
            else:
                answers = None
            if isinstance(answers, str):
                answers = [answers]

            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
                pred_long = pred
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())
            
            task = metadata["task"]
            if task == "number":
                answer = answers[0]
            else:
                answer = string.ascii_uppercase[metadata["answer_idx"]]
            options = metadata.get("options", None)
            score = ego3d_bench_score(answer, pred, task, options)

            if task == "number" and score < 0:
                scores[f"{metadata['category']}_missing"].append(1)
                scores[f"all_missing"].append(1)
            
            else:
                scores[metadata["category"]].append(score)
                if task == "number":
                    assert metadata["category"] in self.NUMBER_CATEGORIES
                    scores["all_mse"].append(score)
                    scores["all"].append(score)
                else:
                    scores["all_acc"].append(score)
                    scores["all"].append(score)
        
        out = {}
        for k in list(self.NUMBER_CATEGORIES):
            out[f"{k}_MSE"] = mean_metric(scores[k])
            out[f"{k}_MISSING"] = sum_metric(scores[f"{k}_missing"])
        for k in list(self.MULTI_CHOICE_CATEGORIES):
            out[f"{k}_ACC"] = mean_metric(scores[k])
        out["All_MSE"] = mean_metric(scores["all_mse"])
        out["All_ACC"] = mean_metric(scores["all_acc"])
        out["All_MISSING"] = sum_metric(scores["all_missing"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores["all"]
            )
        
        return out


class VSIBenchEval(Evaluator):
    MCA_QUESTION_TYPES = [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
    NA_QUESTION_TYPES = [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
    
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        scores = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            if "answer" in metadata:
                answers = metadata["answer"]
            elif "answers" in metadata:
                answers = metadata["answers"]
            else:
                answers = None
            if isinstance(answers, str):
                answers = [answers]

            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
                pred_long = pred
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())
            
            answer = answers[0]
            question_type = metadata["question_type"]
            if question_type in self.MCA_QUESTION_TYPES:
                options = metadata["options"]
                score = muir_bench_mc(answer, pred, options, fallback_to_random=False)
                scores[f"{question_type}_accuracy"].append(score)
            else:
                score = vsi_bench_na_score(pred, answer)
                scores[f"{question_type}_MRA"].append(score)
            scores["all"].append(score)
        
        out = {}
        for k in list(self.MCA_QUESTION_TYPES):
            out[f"{k}_accuracy"] = mean_metric(scores[f"{k}_accuracy"])
        for k in list(self.NA_QUESTION_TYPES):
            out[f"{k}_MRA"] = mean_metric(scores[f"{k}_MRA"])
        
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores["all"]
            )
        return out


class MMIUEval(Evaluator):
    RELATIONSHIPS = [
        '2D-spatial',
        '3D-spatial',
        'Continuous-temporal',
        'Discrete-temporal',
        'High-level-obj-semantic',
        'High-level-sub-semantic',
        'Low-level-semantic',
    ]
    NIMAGES = [
        "num_images<=10",
        "num_images<=20",
        "num_images>20",
    ]


    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer
        scores = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
                pred_long = pred
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())

            options = metadata["options"]
            answer = metadata["answer"]
            # answer = string.ascii_uppercase[metadata["answer_idx"]]
            score = muir_bench_mc(answer, pred, options)

            scores[metadata["relationship"]].append(score)

            nimages = metadata["num_images"]
            if nimages <= 10:
                scores["num_images<=10"].append(score)
            elif nimages <= 20:
                scores["num_images<=20"].append(score)
            else:
                scores["num_images>20"].append(score)
            scores["all"].append(score)

        out = {}
        for k in self.RELATIONSHIPS:
            out[k] = mean_metric(scores[k])
        for k in self.NIMAGES:
            out[k] = mean_metric(scores[k])
        out["all"] = mean_metric(scores["all"])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores["all"]
            )
        return out


class VixMoPointCountEval(Evaluator):
    SUBSETS = ["object", "animal", "action/event"]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        responses = predictions["predictions_text"]
        vocab = tokenizer
        all_scores = defaultdict(list)
        per_category_scores = defaultdict(list)
        gt_counts_per_device = []

        all_pred_triplets = []
        all_gt_triplets = []
        # per_subset_accuracy = {subset: [] for subset in self.SUBSETS}
        per_subset_accuracy = defaultdict(list)
        for ex_ix, pred_seq in enumerate(responses):
            metadata = metadatas[ex_ix]

            pred_int = None
            original_pred = pred_seq.strip()
            pred = original_pred.lower().rstrip(".").strip()

            try:
                pred_int = int(pred)
            except Exception:
                pred_int = None

                # Parse out the int for point and count data
                try:
                    pattern = r'(?:a total of|the total number(?:\s+of\s+.+?)?\s+is:)\s*(\d+)\.?'
                    match = re.search(pattern, pred, re.IGNORECASE)
                    if match:
                        pred_int = int(match.group(1))
                except Exception as e:
                    logging.warning(f"Failed extracting pred_int from {pred} with error - {e}")

                if pred_int is None:
                    match = re.match(".*\bnone\b.*", pred)
                    if match:
                        pred_int = 0

            gt = None
            try:
                gt = int(metadata["count"])
            except Exception as e:
                logging.warning(f"Failed extracting gt count with error - {e}")
                continue

            gt_points = None
            try:
                if "points" in metadata:
                    gt_points = list(metadata["points"])
            except Exception as e:
                logging.warning(f"Failed extracting gt points with error - {e}")

            gt_timestamps = None
            gt_triplets = []
            video_w, video_h = metadata['image_size'] if 'image_size' in metadata else (None, None)
            try:
                if "timestamps" in metadata:
                    gt_timestamps = list(metadata["timestamps"])
                    for i, gt_ts in enumerate(gt_timestamps):
                        points = gt_points[i]
                        gt_triplets.extend([(gt_ts, point["x"] / 100 * video_w, point["y"] / 100 * video_h) for point in points])
                    all_gt_triplets.append(gt_triplets)
            except Exception as e:
                all_gt_triplets.append([])
                logging.warning(f"Failed extracting gt timestamps with error - {e}")

            if pred_int is None:
                correct, close, valid = 0, 0, False
            else:
                correct = gt == pred_int
                margin = 1 + math.floor(0.05 * gt)
                close = abs(gt - pred_int) <= margin
                valid = True
            all_scores["close"].append(close)
            all_scores["valid"].append(valid)
            all_scores["correct"].append(correct)
            per_category_scores[int(gt)].append(correct)
            gt_counts_per_device.append(int(gt))
            per_subset_accuracy[metadata["subset"]].append(correct)
            if self.n_to_log:
                pred_times_and_points = extract_multi_image_points(pred, image_w=video_w, image_h=video_h)
                if len(pred_times_and_points) == 0:
                    logging.warning(f"Failed extracting pred points and timestamps from {pred} with error")
                all_pred_triplets.append(pred_times_and_points)
        num_examples_per_device = torch.tensor(len(responses), dtype=torch.int32, device=torch.device("cuda"))
        num_examples = torch.zeros(get_world_size(), dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(num_examples, num_examples_per_device)
        max_num_examples = num_examples.detach().cpu().max().item()
        gt_counts_per_device = torch.tensor(gt_counts_per_device, dtype=torch.int32, device=torch.device("cuda"))
        gt_counts_per_device = torch.cat(
            [gt_counts_per_device, torch.full((max_num_examples - len(responses),), -1, dtype=torch.int32, device=torch.device("cuda"))],
            dim=0,
        )
        gt_counts = torch.zeros(get_world_size() * max_num_examples, dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(gt_counts, gt_counts_per_device)
        gt_counts = gt_counts.detach().cpu().numpy()
        gt_counts = np.sort(np.unique(gt_counts[gt_counts >= 0]))

        out = {}
        for k, v in all_scores.items():
            out[k] = mean_metric(v)

        # for k in gt_counts:
        #     out[f"correct_{k}"] = mean_metric(per_category_scores[k])

        max_cnt = max(gt_counts)
        bins = [0, 5, 10, 15, 20, 25]
        for i in range(len(bins)):
            bin_lower = bins[i]
            if i == len(bins) - 1:
                bin_upper = max_cnt + 1
            else:
                bin_upper = bins[i+1]
            all_scores_in_bin = []
            for j in range(bin_lower, bin_upper):
                if j in per_category_scores:
                    bin_scores = per_category_scores[j]
                    all_scores_in_bin.extend(bin_scores)
            out[f"correct_{bin_lower}_{bin_upper}"] = mean_metric(all_scores_in_bin)

        for subset in self.SUBSETS:
            out[f"acc_{subset}"] = mean_metric(per_subset_accuracy[subset])

        if self.n_to_log:
            per_example_scores = [{k: all_scores[k][i] for k in all_scores} for i in range(len(responses))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                pred_times_and_points=all_pred_triplets,
                gt_times_and_points=all_gt_triplets,
            )
        return out


def is_point_in_tublet(point: Tuple[float, float, float], mask: Dict[float, np.ndarray]) -> bool:
    """
    Check if the point (t, x, y) is within the region defined by the boolean mask.

    Parameters:
    - point (tuple of floats): (t, x, y) coordinates of the point
    - mask (Dict[float, np.ndarray]): Dictionary of boolean masks for each time step
    """
    t, x, y = point
    if t not in mask:
        return False
    return is_point_in_region((x, y), mask[t])

def compute_precision_tublet(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks_list: List[Dict[float, np.ndarray]]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_tublet(preds[i], masks_list[j]):
            cnt += 1
    return cnt / len(preds)

def compute_recall_tublet(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks_list: List[Dict[float, np.ndarray]]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_tublet(preds[i], masks_list[j]):
            cnt += 1
    return cnt / len(masks_list)


class VixMoPointEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):

        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        vocab = tokenizer
        scores = defaultdict(list)
        all_pred_triplets = []
        all_gt_triplets = []
        for ex_ix, pred in enumerate(response_text):
            metadata = metadatas[ex_ix]
            gt_abs_triplets = metadata["gt_abs_triplets"]
            gt_abs_masks = metadata["gt_abs_masks"]
            video_duration = metadata["video_duration"]
            video_h, video_w = metadata["video_height"], metadata["video_width"]
            if isinstance(pred, str):
                pred_abs_triplets = extract_multi_image_points(pred, image_w=video_w, image_h=video_h)
            else:
                assert isinstance(pred, list), f"Unexpected prediction format: {type(pred)}\n{pred}"
                pred_abs_triplets = pred

            if len(gt_abs_triplets) == 0:
                precision = recall = f1 = float(len(pred_abs_triplets) == 0)
                gt_points = None
            elif len(pred_abs_triplets) == 0:
                precision = recall = f1 = 0.0
            else:
                gt_norm_triplets = normalize_timestamps_and_points(
                    gt_abs_triplets,
                    video_duration=video_duration,
                    video_h=video_h,
                    video_w=video_w,
                    upper_bound=100,
                    num_decimals=1
                )
                pred_norm_triplets = normalize_timestamps_and_points(
                    pred_abs_triplets,
                    video_duration=video_duration,
                    video_h=video_h,
                    video_w=video_w,
                    upper_bound=100,
                    num_decimals=1
                )
                precisions, recalls, f1s = [], [], []
                # bipartite matching with normalized t, x, y coordinates
                dists = cdist(pred_norm_triplets, gt_norm_triplets)
                row_ind, col_ind = linear_sum_assignment(dists)
                # compute precision, recall, f1 with absolute coordinates and gt masks
                precision = compute_precision_tublet(row_ind, col_ind, pred_abs_triplets, gt_abs_masks)
                recall = compute_recall_tublet(row_ind, col_ind, pred_abs_triplets, gt_abs_masks)
                f1 = f1_score(precision, recall)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)
            scores["valid"].append(len(pred_abs_triplets) > 0)

            all_pred_triplets.append(pred_abs_triplets)
            all_gt_triplets.append(gt_abs_triplets)
        out = {}

        for k, v in scores.items():
            out[k] = mean_metric(v)
        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(response_text))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                pred_times_and_points=all_pred_triplets,
                gt_times_and_points=all_gt_triplets,
            )
        return out


class PointBenchEval(Evaluator):
    CATEGORIES = ["affordable", "counting", "reasoning", "spatial", "steerable"]

    @staticmethod
    def is_point_in_mask(x, y, mask, img_width, img_height):
        """Check if a point is inside the mask."""
        # Unpack point (x, y format in pixel coordinates)
        pixel_x = int(x)
        pixel_y = int(y)
        if pixel_y < 0 or pixel_y >= img_height or pixel_x < 0 or pixel_x >= img_width:
            return False
        if pixel_y >= mask.shape[0] or pixel_x >= mask.shape[1]:
            # In case the mask has a different shape then the image, which seems to be due
            # to an error in PointBench masks
            return False
        return mask[pixel_y, pixel_x]

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        scores = []
        category_scores = {cat: [] for cat in self.CATEGORIES}
        for ix, metadata in enumerate(metadatas):
            image_w, image_h = metadata["image_size"]
            mask = metadata["mask"]
            points = _extract_image_points(predictions, ix, image_w, image_h)
            mask_h, mask_w = mask.shape[:2]
            if mask_h != image_h or mask_w != image_w:
                logging.warning(f"Mask and image have different shapes: {(mask_w, mask_h)} {(image_w, image_h)}")
            if len(points) == 0:
                points_in_mask = False
            else:
                points_in_mask = True
                for x, y in points:
                    if not self.is_point_in_mask(x, y, mask, image_w, image_h):
                        points_in_mask = False
                        break
            scores.append(points_in_mask)
            category_scores[metadata["category"]].append(points_in_mask)
        return {k: mean_metric(v) for k, v in category_scores.items()}


class ScreenSpotEvaluator:
    kinds = ("desktop", "web", "mobile")
    types = ["icon", "text"]

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        scores = {}
        for k in self.kinds:
            scores[k] = []
            for t in self.types:
                scores[f"{k}-{t}"] = []
        for ix, (metadata, response) in enumerate(zip(metadatas, response_text)):
            model_points = _extract_image_points(predictions, ix, *metadata["image_size"])
            if len(model_points) == 0:
                acc = 0
            else:
                x, y = model_points[0]
                bbox = metadata["bbox"]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                acc = (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])
            scores[f"{metadata['kind']}"].append(acc)
            scores[f"{metadata['kind']}-{metadata['data_type']}"].append(acc)
        return {k: mean_metric(v) for k, v in scores.items()}


class ScreenSpotProEvaluator:
    CATEGORIES = {
        "Creative": [
            "blender",
            "davinci",
            "fruitloops",
            "illustrator",
            "photoshop",
            "premiere",
            "unreal_engine"
        ],
        "Office": [
            "excel",
            "powerpoint",
            "word"
        ],
        "Dev": [
            "android_studio",
            "pycharm",
            "quartus",
            "vmware",
            "vscode"
        ],
        "CAD": [
            "autocad",
            "inventor",
            "solidworks",
            "vivado"
        ],
        "Scientific": [
            "eviews",
            "matlab",
            "origin",
            "stata"
        ],
        "OS": [
            "linux_common",
            "macos_common",
            "windows_common"
        ]
    }

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        scores = {
            "overall": [],
        }
        for cat, subcats in self.CATEGORIES.items():
            scores[cat] = []
            for subcat in subcats:
                scores[f"{cat}-{subcat}"] = []
        for ix, (metadata, response) in enumerate(zip(metadatas, response_text)):
            model_points = _extract_image_points(predictions, ix, *metadata["image_size"])
            if len(model_points) == 0:
                acc = 0
            else:
                x, y = model_points[0]
                bbox = metadata["bbox"]
                acc = (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])
            scores[f"{metadata['group']}-{metadata['application']}"].append(acc)
            scores[f"{metadata['group']}"].append(acc)
            scores[f"overall"].append(acc)
        return {k: mean_metric(v) for k, v in scores.items()}


class OsWorldGEvaluator:

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        response_text = predictions["predictions_text"]
        prompt_text = predictions["prompts_text"]
        scores = {
            "overall": [],
        }
        for ix, (metadata, response) in enumerate(zip(metadatas, response_text)):
            model_points = _extract_image_points(predictions, ix, *metadata["image_size"])
            if len(model_points) == 0:
                acc = 0
            else:
                x, y = model_points[0]
                bbox = metadata["box_coordinates"]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                acc = (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])
            scores[f"overall"].append(acc)
        return {k: mean_metric(v) for k, v in scores.items()}
