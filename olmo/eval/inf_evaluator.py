"""Class to evaluate models based on their generation outputs"""
import dataclasses
import itertools
import logging
import time
from collections import defaultdict
from typing import List, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics
import wandb
from tqdm import tqdm

from .evaluators import (
    HtmlTable, CountEval, PointCountEval, ClockEval, VqaEval,
    SavePredictions, MathVistaEval, PointingEval,
    TempCompassEval, VideoMMEEval, MLVUGenEval, LongVideoBenchEval, LongVideoBenchCaptionEval,
    VinogroundEval, VixMoCaptionEval, QVHighlightsEval,
    TomatoEval, TemporalBenchEval, Dream1KCaptionEval, MMEVideoOCREval, VideoHallucerEval,
    MMIUEval, LVBenchEval, MulSetEval, Ego3dBenchEval, VSIBenchEval,
    VideoObjectTrackingEval, VixMoPointCountEval, VixMoPointEval,
    PointBenchEval, ScreenSpotProEvaluator, ScreenSpotEvaluator, OsWorldGEvaluator
)
from .open_ended_qa_eval import OpenQaEvaluator
from ..config import BaseConfig
from ..data.data_loader import DataLoaderConfig
from ..nn.beam_search import SamplingConfig, TopKSampler, TopPSampler, MultinomialSampler, \
    TopKTopPSampler, RepeatedNGramBlockingConstraint, RepetitionPenaltyConstraint, \
    FrequencyPenaltyConstraint
from ..torch_util import (
    get_global_rank,
    get_world_size,
    move_to_device, barrier,
)
from ..util import flatten_list

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InfEvaluator:
    """
    Evaluates the text outputs from a model on a task
    """
    metrics: List

    def __call__(self, predictions, example_metadata, tokenizer, device, step=None, **kwargs):
        inf_metrics = {}
        log.info("Computing metrics...")
        for metric in self.metrics:
            results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer, **kwargs)
            for k in results:
                if k in inf_metrics:
                    log.warning(f"Metric {k} had multiple values")
            inf_metrics.update(results)

        log.info("Aggregating metrics...")
        resolved_metrics = {}
        # sort so metrics are iterated on in the same order on all devices
        for k in sorted(inf_metrics):
            v = inf_metrics[k]
            if isinstance(v, float):
                # Trust the Evaluator writer to provide aggregated metrics
                resolved_metrics[k] = v
            elif isinstance(v, torchmetrics.Metric):
                resolved_metrics[k] = v.to(device).compute().item()
            elif isinstance(v, HtmlTable):
                # Special case, we aggregate table rows from all devices to ensure we can always
                # have enough rows to show even if each device only eval-ed a few examples
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    all_rows = flatten_list([x.rows for x in all_predictions])
                    resolved_metrics[k] = wandb.Html(HtmlTable(all_rows).get_html())
                else:
                    dist.gather_object(v, None)
            elif isinstance(v, List):
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    resolved_metrics[k] = []
                    for pred in all_predictions:
                        resolved_metrics[k] += pred
                else:
                    dist.gather_object(v, None)
            else:
                raise ValueError(f"Metric {v} not understood, must be aggregated between devices and of type float|List|HtmlTable|torchmetrics.Metric")

        # Some metrics need to some kind of more complex aggregation that cannot be done on the
        # individual worker, we hack those special cases in here
        for metric in self.metrics:
            if isinstance(metric, VSIBenchEval):
                resolved_metrics["object_rel_direction_accuracy"] = sum(
                    [
                        resolved_metrics[f"{k}_accuracy"]
                        for k in [
                            "object_rel_direction_easy",
                            "object_rel_direction_medium",
                            "object_rel_direction_hard",
                        ]
                    ]
                ) / 3.
                acc_types = [
                    "object_rel_direction",
                    "object_rel_distance",
                    "route_planning",
                    "obj_appearance_order",
                ]
                overall = [
                    resolved_metrics[f"{k}_MRA"]
                    for k in metric.NA_QUESTION_TYPES
                ]
                overall += [
                    resolved_metrics[f"{k}_accuracy"]
                    for k in acc_types
                ]
                resolved_metrics["overall"] = np.mean(overall)
            elif isinstance(metric, MulSetEval):
                resolved_metrics["overall"] = np.mean(
                    [
                        resolved_metrics[k]
                        for k in list(metric.TASKS.values())
                    ]
                )
            elif isinstance(metric, Ego3dBenchEval):
                # Compute RMSE
                for k in list(resolved_metrics.keys()):
                    if "MSE" in k:
                        resolved_metrics[k.replace("MSE", "RMSE")] = np.sqrt(resolved_metrics[k])
            elif isinstance(metric, (PointBenchEval,)):
                resolved_metrics["average"] = sum(resolved_metrics.get(cat) for cat in PointBenchEval.CATEGORIES) / len(PointBenchEval.CATEGORIES)
            elif isinstance(metric, (CountEval, PointCountEval, VixMoPointCountEval)):
                # Counting has a macro-score that should be computed once we have
                # scores from all devices
                counting_scores = {k: resolved_metrics[k] for
                                   k in list(resolved_metrics.keys()) if k.startswith("correct_")}
                resolved_metrics["per_category_average"] = np.mean(list(counting_scores.values()))
            elif isinstance(metric, MLVUGenEval):
                # MLVU has a macro-score that should be computed once we have
                # scores from all devices
                mlvu_sub_scene_scores = {k: resolved_metrics[k] for
                                         k in list(resolved_metrics.keys()) if k.startswith("sub_scene_")}
                resolved_metrics["sub_scene_total"] = np.sum(list(mlvu_sub_scene_scores.values()))
                mlvu_summary_scores = {k: resolved_metrics[k] for
                                      k in list(resolved_metrics.keys()) if k.startswith("summary_")}
                resolved_metrics["summary_total"] = np.sum(list(mlvu_summary_scores.values()))
                resolved_metrics["mlvu_gen_total"] = np.mean([resolved_metrics["sub_scene_total"], resolved_metrics["summary_total"]])
            elif isinstance(metric, TemporalBenchEval) and get_global_rank() == 0:
                type_video_scores = defaultdict(lambda: defaultdict(list))
                for cat, idx, score in zip(
                    resolved_metrics.pop("kind"),
                    resolved_metrics.pop("idx"),
                    resolved_metrics.pop("score"),
                ):
                    type_video_scores[cat][idx].append(score)
                total_score = 0
                for cat, video_scores in type_video_scores.items():
                    cat_score = sum(all(v) for v in video_scores.values())
                    resolved_metrics[cat] = cat_score / len(video_scores)
                    resolved_metrics[cat+"_count"] = len(video_scores)
                resolved_metrics["all"] = total_score / sum(len(x) for x in type_video_scores.values())
            elif isinstance(metric, VinogroundEval) and get_global_rank() == 0:
                score_matrix = torch.zeros((500, 4), dtype=torch.bool)
                for i in range(1000):
                    text_score = resolved_metrics["text"][i]
                    text_idx = resolved_metrics["text_idx"][i]
                    id, pos_neg = text_idx.split('_')
                    score_matrix[int(id), pos_neg=="neg"] = text_score

                    video_score = resolved_metrics["video"][i]
                    video_idx = resolved_metrics["video_idx"][i]
                    id, pos_neg = video_idx.split('_')
                    score_matrix[int(id), (pos_neg=="neg") + 2] = video_score
                resolved_metrics["text"] = (score_matrix[:, 0] & score_matrix[:, 1]).float().mean().item()
                resolved_metrics["video"] = (score_matrix[:, 2] & score_matrix[:, 3]).float().mean().item()
                resolved_metrics["group"] = (score_matrix[:, 0] & score_matrix[:, 1] & score_matrix[:, 2] & score_matrix[:, 3]).float().mean().item()
        return resolved_metrics


@dataclasses.dataclass
class EvaluatorConfig(BaseConfig):
    """Config for `Evaluator` objects that compute metrics"""

    n_to_log: int = 10
    """Num examples to log to console"""

    num_wandb_examples: int = 0
    """Num examples to log to Wandb as a HTML table"""

    save_predictions: Optional[str] = "_default"  # saves with default name to checkpoint dir
    """Where to save predictions files"""

    save_tokens: bool = False
    """If save predictions, should the tokens be saved"""

    vqa_eval: str = ''
    """name(s) of VQA-style eval to run, can be a comma seperated list"""

    # Other individual types of eval
    pointing_eval: bool = False
    point_bench_eval: bool = False
    count_eval: bool = False
    point_count_eval: bool = False
    clock_eval: bool = False
    clock_bench_eval: bool = False # Clock reading benchmark, coco/openimg/movies
    math_vista_eval: bool = False
    temp_compass_eval: str = ''
    """TempCompass tasks to run evaluation on, either one of the tasks or 'all'"""
    temp_compass_disable_api: bool = False
    """Whether not to use ChatGPT evaluation for TempCompass"""
    video_mme_eval: str = ''
    mme_videoocr_eval: bool = False
    """VideoMME tasks to run evaluation on, either one of the tasks or 'all'"""
    mlvu_gen_eval: bool = False
    lvbench_eval: bool = False
    long_video_bench_eval: bool = False
    video_hallucer: bool = False
    long_video_bench_caption_eval: bool = False
    vinoground_eval: bool = False
    vixmo_caption_eval: bool = False
    vixmo_caption_eval2: bool = False
    dream1k_caption_eval: bool = False
    vixmo_point_count_eval: bool = False
    vixmo_point_eval: bool = False

    """ Video Object Tracking evaluation """
    video_object_tracking_eval: str = '' # path with object tracking predictions
    video_single_point_prediction: str='' # path with single point predicitons
    video_point_tracking_eval: str = ''
    """Video pointing task name to run evaluation on (e.g., 'mevis_point_track_per_frame_fps_6_sample_fps_1')"""
    
    """Whether to run RefExp evaluation"""
    refexp_eval: bool = False
    """Whether to run COCO captioning evaluation, use CIDEr score"""
    qv_highlights_eval: bool = False
    tomato: bool = False
    temporal_bench: bool = False
    open_qa_eval: bool = False
    mmiu_eval: bool = False
    mulset_eval: bool = False
    ego3d_bench_eval: bool = False
    vsi_bench_eval: bool = False
    os_worldg_evaluation: bool = False
    screen_spot_evaluator: bool = False
    screen_spot_pro_evaluator: bool = False

    def build(self, default_save_dir=None) -> InfEvaluator:
        evaluators = []
        save_predictions = self.save_predictions
        if save_predictions == "_default":
            if default_save_dir is None:
                logging.info(f"save_predictions is \"default\" but no default "
                             f"save dir set so predictions will not be saved")
            save_predictions = default_save_dir
        if save_predictions:
            evaluators.append(SavePredictions(
                save_predictions,
                log_examples=self.n_to_log,
                save_tokens=self.save_tokens
            ))

        if self.vqa_eval:
            evaluators.append(VqaEval(self.vqa_eval.split(","), self.num_wandb_examples))
        if self.tomato:
            evaluators.append(TomatoEval(self.num_wandb_examples))
        if self.temporal_bench:
            evaluators.append(TemporalBenchEval(self.num_wandb_examples))
        elif self.clock_eval:
            evaluators.append(ClockEval(self.num_wandb_examples))
        elif self.clock_bench_eval:
            evaluators.append(ClockEval(self.num_wandb_examples, is_test=True))
        elif self.math_vista_eval:
            evaluators.append(MathVistaEval(self.num_wandb_examples))
        elif self.point_count_eval:
            evaluators.append(PointCountEval(self.num_wandb_examples))
        elif self.count_eval:
            evaluators.append(CountEval(self.num_wandb_examples))
        elif self.point_bench_eval:
            evaluators.append(PointBenchEval(self.num_wandb_examples))
        elif self.temp_compass_eval:
            evaluators.append(TempCompassEval(self.temp_compass_eval, self.temp_compass_disable_api, self.num_wandb_examples))
        elif self.video_mme_eval:
            evaluators.append(VideoMMEEval(self.video_mme_eval, self.num_wandb_examples))
        elif self.mlvu_gen_eval:
            evaluators.append(MLVUGenEval(self.num_wandb_examples))
        elif self.long_video_bench_eval:
            evaluators.append(LongVideoBenchEval(self.num_wandb_examples))
        if self.pointing_eval:
            evaluators.append(PointingEval(self.num_wandb_examples))
        if self.lvbench_eval:
            evaluators.append(LVBenchEval(self.num_wandb_examples))
        if self.long_video_bench_caption_eval:
            evaluators.append(LongVideoBenchCaptionEval(self.num_wandb_examples))
        if self.vinoground_eval:
            evaluators.append(VinogroundEval(self.num_wandb_examples))
        if self.vixmo_caption_eval:
            evaluators.append(VixMoCaptionEval(self.num_wandb_examples))
        if self.vixmo_caption_eval2:
            evaluators.append(VixMoCaptionEval(self.num_wandb_examples, version='v2'))
        if self.dream1k_caption_eval:
            evaluators.append(Dream1KCaptionEval(self.num_wandb_examples))
        if self.mme_videoocr_eval:
            evaluators.append(MMEVideoOCREval(self.num_wandb_examples))
        elif self.video_object_tracking_eval:
            evaluators.append(VideoObjectTrackingEval(self.num_wandb_examples))
        if self.qv_highlights_eval:
            evaluators.append(QVHighlightsEval(self.num_wandb_examples))
        if self.open_qa_eval:
            evaluators.append(OpenQaEvaluator(self.num_wandb_examples))
        if self.video_hallucer:
            evaluators.append(VideoHallucerEval(self.num_wandb_examples))
        if self.mmiu_eval:
            evaluators.append(MMIUEval(self.num_wandb_examples))
        if self.mulset_eval:
            evaluators.append(MulSetEval(self.num_wandb_examples))
        if self.ego3d_bench_eval:
            evaluators.append(Ego3dBenchEval(self.num_wandb_examples))
        if self.vsi_bench_eval:
            evaluators.append(VSIBenchEval(self.num_wandb_examples))
        if self.vixmo_point_count_eval:
            evaluators.append(VixMoPointCountEval(self.num_wandb_examples))
        if self.vixmo_point_eval:
            evaluators.append(VixMoPointEval(self.num_wandb_examples))
        elif self.os_worldg_evaluation:
            evaluators.append(OsWorldGEvaluator())
        elif self.screen_spot_evaluator:
            evaluators.append(ScreenSpotEvaluator())
        elif self.screen_spot_pro_evaluator:
            evaluators.append(ScreenSpotProEvaluator())
        else:
            pass
        return InfEvaluator(evaluators)


@dataclasses.dataclass
class InfDatasetEvaluator:
    """Evaluates a model on a dataset"""
    label: str
    dataloader: Any
    evaluator: InfEvaluator
    n_steps: int
    max_new_tokens: int = 448
    console_log_interval: Optional[int] = None
    sampling_parameters: Optional[SamplingConfig] = None

    def run(self, model, device, autocast_precision, is_distributed, pbar=False, logger=None):
        eval_dataloader = self.dataloader
        eval_it = iter(eval_dataloader)
        n_steps = self.n_steps
        if n_steps is not None and 0 <= n_steps < len(self.dataloader):
            eval_it = itertools.islice(eval_it, 0, n_steps)
            total_steps = n_steps
        else:
            total_steps = len(eval_dataloader)

        constraints = []
        if self.sampling_parameters is None:
            sampler = None
        else:
            sampling = self.sampling_parameters
            if sampling.top_k is None and sampling.top_p == 1 and sampling.temperature == 0 and not sampling.ngram_size:
                sampler = None
            else:
                sampler = TopKTopPSampler(p=sampling.top_p, k=sampling.top_k, temperature=sampling.temperature)
            if sampling.ngram_size:
                constraints.append(RepeatedNGramBlockingConstraint(ngram_size=sampling.ngram_size))
            if sampling.repetition_penalty:
                constraints.append(RepetitionPenaltyConstraint(penalty=sampling.repetition_penalty))
            if sampling.frequency_penalty:
                constraints.append(FrequencyPenaltyConstraint(penalty=sampling.frequency_penalty))
        all_metadata = []
        predictions = defaultdict(list)
        done_init = False
        tok = model.config.build_tokenizer()
        pbar = pbar and get_global_rank() == 0
        for eval_step, batch in enumerate(tqdm(eval_it, total=total_steps, ncols=100, disable=not pbar)):
            if logger and eval_step % logger.log_interval == 0:
                logger.log_evaluation(self.label, eval_step, total_steps)
            if "metadata" in batch:
                batch_metadata = batch.pop("metadata")
            else:
                # Handle old-style data that used metadata/ prefix instead
                metadata = {k: batch.pop(k) for k in list(batch) if k.startswith("metadata/")}
                batch_metadata = []
                for i in range(len(batch["input_ids"])):
                    converted = {}
                    for k, v in metadata.items():
                        if isinstance(v[i], bytes):
                            converted[k] = v[i].decode("utf-8")
                        else:
                            converted[k] = v[i].tolist()
                    batch_metadata.append(converted)
            batch_inference = move_to_device(batch, device)
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    olmo_gen_output = model.generate(
                        batch=batch_inference,
                        max_steps=self.max_new_tokens,
                        sampler=sampler,
                        constraints=constraints,
                        is_distributed=is_distributed
                    )
            input_tokens = olmo_gen_output.token_ids[:, 0].detach().cpu().numpy()
            prompt_tokens = batch_inference["input_ids"].detach().cpu().numpy()
            prediction_text = [tok.decode(x[x >= 0]) for x in input_tokens]
            pred = {
                "predictions": input_tokens, # beam size of 1
                "prompts": prompt_tokens,
                "predictions_text": prediction_text,
                "prompts_text": [tok.decode(x[x >= 0]) for x in prompt_tokens],
            }
            if olmo_gen_output.token_target_ids is not None:
                points = []
                for text, point_indices, metadata in zip(prediction_text, olmo_gen_output.token_target_ids, batch_metadata):
                    points.append(model.config.token_ids_to_coordinates(text, point_indices, metadata))
                pred["points"] = points

            valid_ixs = [i for i, md in enumerate(batch_metadata) if md.get("valid", True)]
            all_metadata += [batch_metadata[i] for i in valid_ixs]
            for k, v in pred.items():
                for ix in valid_ixs:
                    predictions[k].append(v[ix])

            # Log to console.
            if self.console_log_interval and not pbar:
                if eval_step + 1 == n_steps or (eval_step + 1) % self.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{total_steps}]")

        barrier()
        tokenizer = model.config.build_tokenizer()
        if logger:
            logger.log_evaluation(self.label, total_steps, total_steps)
        metrics = self.evaluator(predictions, all_metadata, tokenizer, device)
        return metrics


@dataclasses.dataclass
class InfDatasetEvaluatorConfig(BaseConfig):
    """Configuration for an inference evaluator"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig)
    """Evaluator to compute metrics from the generated outputs"""

    max_new_tokens: int = 448
    """Max number of tokens to generate"""

    device_batch_size: int = 4
    """Batch size"""

    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""

    include_image: bool = False
    """Include image in the metadata"""

    def build_dataset_evaluator(
        self,
        model_config,
        mesh,
        default_save_dir,
        device,
    ) -> InfDatasetEvaluator:
        assert mesh is None, "Mesh not supported for inference for now"
        global_batch_size = self.device_batch_size * get_world_size()
        if self.max_examples and self.max_examples > 0:
            max_steps = max(self.max_examples // global_batch_size, 1)
        elif self.subset_num_batches:
            max_steps = self.subset_num_batches
        else:
            max_steps = None

        eval_loader = self.data.build_eval_dataloader(
            model_config=model_config,
            batch_size=self.device_batch_size,
            mesh=mesh,
            for_inference=True,
            pad_batches=True,
            max_steps_for_padding=max_steps,
            include_image=self.include_image,
        )
        if self.max_examples is not None:
            num_batches = self.max_examples // self.device_batch_size*get_world_size()
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return InfDatasetEvaluator(
            label=self.label,
            dataloader=eval_loader,
            evaluator=self.evaluator.build(default_save_dir),
            n_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            console_log_interval=self.console_log_interval,
            sampling_parameters=self.sampling
        )
