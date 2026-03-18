"""
Script for evaluating Molmo2 models
"""

import argparse
import logging
from dataclasses import replace
from typing import cast

import torch.cuda
from omegaconf import OmegaConf

from olmo.eval.eval_utils import get_evaluation
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo_point.molmo_point import MolmoPointConfig
from olmo.train.trainer_config import FSDPConfig, FSDPPrecision
from olmo.models.model import FSDPWrapStrategy
from olmo.util import (
    clean_opt,
    prepare_torchrun_environment, select_checkpoint, resource_path, )
from olmo.model_configs import DEBUG_MOLMO2

from olmo.eval.model_evaluator import DatasetEvaluatorConfig, EvalConfig
import omegaconf.omegaconf as om

log = logging.getLogger(__name__)


IMAGE_TASKS = [
    "coco_2014_vqa_8192",
    "text_vqa",
    "chart_qa",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "mmmu_test",
    "real_world_qa_no_instruction:test",
    "math_vista_v2",
]

TEST_IMAGE_TASKS = [
    "info_qa:test",
    "doc_qa:test",
    "chart_qa:test",
    "text_vqa",
    "ai2_diagram_v2_mix_transparent:test",
    "math_vista_v2",
    "real_world_qa_no_instruction:test",
    "a_okvqa_mc",
    "a_okvqa_da",
    "mmmu_test",
    "a_okvqa_mc:test",
    "a_okvqa_da:test",
    "countbench_qa:huggingface",
    "pixmo_count_counting:test",
    "coco_2014_vqa:test",
]

VIDEO_POINTING = [
    "vixmo_points_count_clip_63s:val",
    "vixmo_points_point_eval:val",
    "mevis_point_track_per_frame_fps_6_sample_fps_1",
]

TEST_VIDEO_DATASETS = [
    "mlvu_mc_test",
    "perception_test_test",
    "ego_schema_test",
    "motionbench_test",
    "long_video_bench_w_subtitle_test",
]

LONG_VIDEO_SUBTITLES = [
    "video_mme_w_subtitle",
    "long_video_bench_w_subtitle",
    "lvbench",
]

LONG_VIDEO_NO_SUBTITLES = [
    "mlvu_mc",
    "video_mme",
    "long_video_bench_no_subtitle",
    "vixmo_caps_eval2:test",
    "video_eval_pro_mc:test",
]

LONG_VIDEO = LONG_VIDEO_SUBTITLES + LONG_VIDEO_NO_SUBTITLES
TEST_VAL_LONG_VIDEO_SUBTITLE = LONG_VIDEO + ["long_video_bench_w_subtitle:test"]

SHORT_VIDEO = [
    "mvbench",
    "tomato:test",
    "motionbench:validation",
    "temp_compass_internal",
    "perception_test",
    "ego_schema",
    "nextqa_mc:test",
]

USE_QUERY_SEL = [x.split(":")[0] for x in SHORT_VIDEO + LONG_VIDEO + VIDEO_POINTING]


MULTI_IMAGE_TASKS = [
    "muir_bench:test",
    "mmiu:test",
    "blink:validation",
]

IMAGE_POINTING = [
    "countbench_qa:huggingface",
    "pixmo_count_counting:validation",
    "pointing_eval_v2:test",
    "point_bench:test"
]

TRACKING = [
    "mevis_track_eval_1fps:test",
    "ref_yt_vos_track_eval_1fps:test",
    "ref_davis17_track_eval_1fps:test",
    "reasonvos_track_eval_1fps:test",
    "molmo2_video_track_eval_1fps:test",
]


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Evaluate a model on downstream tasks")
    parser.add_argument("checkpoint", nargs="+",
                        help="Checkpoint to evaluate, should contain a config file and unshared model file")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--seq_len", default=None, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--device_batch_size", default="4b-default")
    parser.add_argument("--eval_name",
                        help="Name to use as a prefix when saving results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--tasks", default="video")

    args, other_args = parser.parse_known_args()
    tasks = []
    loss_eval_tasks = []

    for task_name in args.tasks.split(","):
        if task_name in ["video_pointing"]:
            tasks += VIDEO_POINTING
        elif task_name in ["image_pointing"]:
            tasks += IMAGE_POINTING
        elif task_name in ["short_video"]:
            tasks += SHORT_VIDEO
        elif task_name in ["long_video"]:
            tasks += LONG_VIDEO
        elif task_name in ["test_video"]:
            tasks += TEST_VIDEO_DATASETS
        elif task_name in ["video"]:
            tasks += SHORT_VIDEO
            tasks += LONG_VIDEO
        elif task_name in ["video_no_subtitle"]:
            tasks += SHORT_VIDEO
            tasks += LONG_VIDEO_NO_SUBTITLES
        elif task_name in ["video_subtitle"]:
            tasks += LONG_VIDEO_SUBTITLES
        elif task_name in ["multi_image"]:
            tasks += MULTI_IMAGE_TASKS
        elif task_name in ["single_image"]:
            tasks += IMAGE_TASKS
            tasks += IMAGE_POINTING
            loss_eval_tasks += ["pixmo_ask_model_anything", "pixmo_cap"]
        elif task_name in ["single_image_test"]:
            tasks += TEST_IMAGE_TASKS
        elif task_name in ["tracking"]:
            tasks += TRACKING
        else:
            tasks += [task_name]
    if not tasks:
        raise ValueError()

    inf_evaluators = []
    for task in {k: None for k in tasks}:
        base_config = get_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples,
                                     num_workers=args.num_workers)
        if args.device_batch_size == "4b-default":
            # task-specific default
            if task.startswith("vixmo_points_point_eval"):
                device_batch_size = 1
            elif task in TRACKING:
                device_batch_size = 1
            elif task in IMAGE_TASKS+IMAGE_POINTING:
                device_batch_size = 4
            else:
                device_batch_size = 2
        elif args.device_batch_size == "8b-default":
            if task in TEST_VAL_LONG_VIDEO_SUBTITLE or task in VIDEO_POINTING:
                device_batch_size = 1
            elif task in TRACKING:
                device_batch_size = 1
            elif task in IMAGE_TASKS+IMAGE_POINTING:
                device_batch_size = 4
            else:
                device_batch_size = 2
        else:
            device_batch_size = int(args.device_batch_size)
        eval_config = DatasetEvaluatorConfig(
            label=base_config.label,
            data=replace(base_config.data, pad=None),
            generative_evaluator=replace(
                base_config.evaluator,
                n_to_log=4,
                num_wandb_examples=10,
                save_predictions="_default",
            ),
            sampling=base_config.sampling,
            device_batch_size=device_batch_size,
            subset_num_batches=None,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens or base_config.max_new_tokens,
        )
        inf_evaluators.append(eval_config)
    for task in loss_eval_tasks:
        base_config = get_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples,
                                     num_workers=args.num_workers, for_inference=False)
        eval_config = DatasetEvaluatorConfig(
            label=base_config.label,
            data=replace(base_config.data, pad="to_max" if args.fsdp else None),
            generative_evaluator=None,
            device_batch_size=4,
            subset_num_batches=None,
            max_examples=args.max_examples,
        )
        inf_evaluators.append(eval_config)

    for checkpoint in args.checkpoint:
        log.info(f"*"*40)
        log.info(f"Starting checkpoint {checkpoint}")
        log.info(f"*"*40)
        is_debug = False
        checkpoint_dir = select_checkpoint(checkpoint)
        model_cfg_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")

        model_cfg = MolmoConfig.load(model_cfg_path, key="model", validate_paths=False)
        if isinstance(model_cfg, (Molmo2Config, MolmoPointConfig)):
            if model_cfg.mm_preprocessor.image is not None:
                model_cfg.mm_preprocessor.image.max_crops = max(model_cfg.mm_preprocessor.image.max_crops, 24)
                model_cfg.mm_preprocessor.image.max_images = 20
        elif isinstance(model_cfg, MolmoConfig):
            model_cfg.mm_preprocessor.max_crops = 24
            model_cfg.mm_preprocessor.max_images = 20
        else:
            raise NotImplementedError()
        model_cfg.llm.max_sequence_length = 64000

        cfg = EvalConfig(
            evaluations=inf_evaluators,
            load_path=checkpoint_dir,
            console_log_interval=5,
            beaker_log_interval=5,
            precision="amp_bf16",
            pbar=False,
            eval_name=args.eval_name,
            fsdp=FSDPConfig(fsdp2=True) if args.fsdp else None,
            skip_if_metrics_cached=True,
            save_dir=None,
            model=model_cfg,
            save_to_checkpoint_dir=True,
        )

        config = OmegaConf.create(cfg)
        config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
        cfg = cast(EvalConfig, OmegaConf.to_object(config))
        cfg.build().run()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
