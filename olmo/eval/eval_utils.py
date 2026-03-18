from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.inf_evaluator import EvaluatorConfig, InfDatasetEvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.registry import registry


def get_evaluator(name) -> EvaluatorConfig:
    """Gets the default evaluator for task `name`"""
    if name in [
        "text_vqa",
        "okvqa",
        "coco_2014_vqa",
        "coco_2014_vqa_multi",
        "coco_2014_vqa_8192",
    ]:
        return EvaluatorConfig(vqa_eval="vqa_score")
    elif name.startswith("math_vista"):
        return EvaluatorConfig(math_vista_eval=True)
    elif name == "point_bench":
        return EvaluatorConfig(point_bench_eval=True)
    elif name == "a_okvqa_da":
        return EvaluatorConfig(vqa_eval="a_okvqa_score")
    elif name.startswith("chart_qa"):
        return EvaluatorConfig(
            vqa_eval="relaxed_correctness,scifi_relaxed_correctness,em"
        )
    elif name in ["doc_qa", "info_qa", "st_qa"]:
        return EvaluatorConfig(vqa_eval="ansl,em")
    elif name in ["gqa", "tally_qa"]:
        return EvaluatorConfig(vqa_eval="em")
    elif name in [
        "science_qa",
        "a_okvqa_mc",
        "science_qa_img",
        "ai2_diagram",
        "ai2_diagram_v2",
        "ai2_diagram_v2_transparent",
        "muir_bench_legacy_multiple_choice",
    ]:
        return EvaluatorConfig(vqa_eval="mc")
    elif name in [
        "ai2_diagram_v2_mix_transparent",
        "ai2_diagram_v2_mix_transparent_one_style",
    ]:
        return EvaluatorConfig(vqa_eval="mc_ai2d_opaque,mc_ai2d_transparent")
    elif name.startswith("mmmu"):
        return EvaluatorConfig(vqa_eval="mmmu_score")
    elif name.startswith("countbench_qa") or name.startswith("pixmo_count"):
        return EvaluatorConfig(point_count_eval=True)
    elif name.startswith("real_world_qa"):
        return EvaluatorConfig(vqa_eval="real_world_qa_score")
    elif name == "pixmo_clocks":
        return EvaluatorConfig(clock_eval=True)
    elif name.startswith("pointing_eval"):
        return EvaluatorConfig(pointing_eval=True)
    elif name == "tomato":
        return EvaluatorConfig(tomato=True)
    elif name.startswith("pixmo_ask_model_anything"):
        return EvaluatorConfig(open_qa_eval=True)
    elif "temporal_bench" in name:
        return EvaluatorConfig(temporal_bench=True)
    elif name == "clock_bench":
        return EvaluatorConfig(clock_bench_eval=True)
    elif name in ["countbench_qa"]:
        return EvaluatorConfig(count_eval=True)
    elif (
        "mvbench" in name
        or name in ["llava_video_178k_mc", "motionbench"]
        or any(name.startswith(x) for x in ["mlvu_mc", "video_eval_pro_mc"])
    ):  # expects a single character followed by a dot.
        return EvaluatorConfig(vqa_eval="em_start")
    elif name.startswith("temp_compass"):
        disable_api = "disable_api" in name
        name = name.replace("_disable_api", "")
        task = "_".join(name.split("_")[2:]) if len(name.split("_")) > 2 else "all"
        return EvaluatorConfig(
            temp_compass_eval=task, temp_compass_disable_api=disable_api
        )
    elif name.startswith("video_hallucer"):
        return EvaluatorConfig(video_hallucer=True)
    elif name == "mlvu_gen":
        return EvaluatorConfig(mlvu_gen_eval=True)
    elif name == "ego_schema":
        return EvaluatorConfig(vqa_eval="ego_schema_mc")
    elif name == "perception_test":
        return EvaluatorConfig(vqa_eval="perception_test_mc")
    elif name == "video_mme_w_subtitle":
        return EvaluatorConfig(video_mme_eval="all")
    elif name.startswith("video_mme"):
        duration = "all" if len(name.split("_")) == 2 else name.split("_")[2]
        return EvaluatorConfig(video_mme_eval=duration)
    elif name in [
        "long_video_bench",
        "long_video_bench_no_subtitle",
        "long_video_bench_w_subtitle",
    ]:
        return EvaluatorConfig(long_video_bench_eval=True)
    elif name == "nextqa_mc":
        return EvaluatorConfig(vqa_eval="nextqa_mc")
    elif name in [
        "muir_bench",
        "muir_bench_legacy_short_answer",
        "muir_bench_legacy_answer_first",
        "muir_bench_legacy_answer_last",
    ]:
        return EvaluatorConfig(vqa_eval="muir_bench_mc")
    elif name in [
        "mmsi_bench",
        "mmsi_bench_legacy_short_answer",
        "mmsi_bench_legacy_answer_first",
        "mmsi_bench_legacy_answer_last",
    ]:
        return EvaluatorConfig(vqa_eval="muir_bench_mc")
    elif name in [
        "mantis_eval",
        "mantis_eval_multi_choice",
        "mantis_eval_short_answer",
    ]:
        return EvaluatorConfig(vqa_eval="mantis_eval_mc")
    elif name in [
        "mantis_eval_legacy",
        "mantis_eval_legacy_multi_choice",
        "mantis_eval_legacy_short_answer",
    ]:
        return EvaluatorConfig(vqa_eval="mantis_eval_mc")
    elif name in [
        "mmiu",
        "mmiu_legacy_short_answer",
        "mmiu_legacy_answer_first",
        "mmiu_legacy_answer_last",
    ]:
        return EvaluatorConfig(mmiu_eval=True)
    elif name in ["mulset"]:
        return EvaluatorConfig(mulset_eval=True)
    elif name in ["ego3d_bench"]:
        return EvaluatorConfig(ego3d_bench_eval=True)
    elif name in ["blink"]:
        return EvaluatorConfig(vqa_eval="muir_bench_mc")
    elif name.startswith("vsi_bench"):
        return EvaluatorConfig(vsi_bench_eval=True)
    elif any(task in name for task in ["predicted_single_point_track_per_frame"]):
        return EvaluatorConfig(video_single_point_prediction=name)
    elif any(
        task in name
        for task in [
            "point_track_per_frame",
            "point_ground_start_end",
            "single_point_track_per_frame",
        ]
    ):
        return EvaluatorConfig(video_object_tracking_eval=name)
    elif any(task in name for task in ["track_eval"]):
        return EvaluatorConfig(video_object_tracking_eval="point_track_per_frame")
    elif any(task in name for task in ["tap_davis"]):
        return EvaluatorConfig(
            video_point_tracking_eval="point_track_all_frames_with_occlusion"
        )
    elif name.startswith("predicted_point_track"):
        # For two-stage pointing evaluation - using predicted points from stage 1
        return EvaluatorConfig(video_point_tracking_eval="single_point_track_per_frame")
    elif name in [
        "dense_caption_eval",
        "user_qa",
        "vqa_v2_test",
        "intern_vid",
        "ego_schema_test",
        "molmo2_human_eval",
        "perception_test_test",
        "long_video_bench_w_subtitle_test",
        "motionbench_test",
    ] or name.startswith("molmo2_human_eval")  or name.startswith("vixmo_ab_test"):
        # No metrics, but still save prediction file
        return EvaluatorConfig()
    elif name == "lvbench":
        return EvaluatorConfig(lvbench_eval=True)
    elif name == "long_video_bench_caption":
        return EvaluatorConfig(long_video_bench_caption_eval=True)
    elif name == "vinoground":
        return EvaluatorConfig(vinoground_eval=True)
    elif name.startswith("vixmo_caps_eval"):
        if name == "vixmo_caps_eval2":
            return EvaluatorConfig(vixmo_caption_eval2=True)
        else:
            return EvaluatorConfig(vixmo_caption_eval=True)
    elif name == "dream1k":
        return EvaluatorConfig(dream1k_caption_eval=True)
    elif name.startswith("mme_videoocr"):
        return EvaluatorConfig(mme_videoocr_eval=True)
    elif name == "qv_highlights":
        return EvaluatorConfig(qv_highlights_eval=True)
    elif name in [
        "vixmo_points_count",
        "vixmo_points_count_clip_63s",
        "academic_points_count_clip_63s",
        "lvvis_count_clip_63s",
        "burst_count_clip_63s",
    ]:
        return EvaluatorConfig(vixmo_point_count_eval=True)
    elif name == "vixmo_points_point_eval":
        return EvaluatorConfig(vixmo_point_eval=True)
    elif name.startswith("screen_spot_v2"):
        return EvaluatorConfig(screen_spot_evaluator=True)
    elif name.startswith("os_worldg"):
        return EvaluatorConfig(os_worldg_evaluation=True)
    elif name.startswith("screen_spot_pro"):
        return EvaluatorConfig(screen_spot_pro_evaluator=True)
    elif f"evaluator/{name}" in registry.list():
        return registry.make(f"evaluator/{name}")
    else:
        raise NotImplementedError(name)


def get_default_max_tokens(name):
    if name == "dense_caption_eval":
        return 448
    if name.startswith("vixmo_ab_test"):
        return 2048
    if name.startswith("molmo2_human_eval"):
        return 4096
    elif name.startswith("named_entity"):
        max_new_tokens = 256
    elif name.startswith("pixmo_ask_model_anything"):
        max_new_tokens = 384
    elif name == "math_vista_demo":
        max_new_tokens = 384
    elif name in [
        "chart_qa_scifi",
        "chart_qa_ex",
        "chart_qa_exp",
        "chart_qa_prompting_explanation",
    ] or name.endswith("_demo"):
        max_new_tokens = 256
    elif name.startswith("user_questions_for_elo"):
        max_new_tokens = 768  # Can have counts of 20+ so make sure there is room
    elif name.startswith("screen_spot") or name.startswith("os_worldg"):
        max_new_tokens = 128
    elif name == "point_bench":
        max_new_tokens = 256
    elif name in ["pointing_eval", "pointing_eval_v2", "pointing"]:
        max_new_tokens = 192  # 192 is enought for counts <=10 in the point tag format
    elif "countbench_qa" in name or "pixmo_count" in name:
        max_new_tokens = 192
    elif name == "android_control_hl_cot":
        max_new_tokens = 64
    elif name.startswith("android_control") or name.startswith("vsi_bench"):
        max_new_tokens = 16
    elif (
        name == "llava_video_178k_oe"
        or name == "llava_video_178k_cap"
        or name.startswith("temp_compass")
        or name == "mlvu_gen"
        or name == "mme_videoocr"
    ):
        max_new_tokens = 192
    elif (
        "refc" in name
        or "mvbench" in name
        or name in ["llava_video_178k_mc", "video_eval_pro_mc", "mme_videoocr_mc"]
    ):
        max_new_tokens = 32
    elif name == "ego_schema" or name in [
        "muir_bench",
        "muir_bench_legacy_short_answer",
    ]:
        max_new_tokens = 32
    elif name == "long_video_bench_caption":
        max_new_tokens = 768
    elif name.startswith("vixmo_caps_eval"):
        max_new_tokens = 2048
    elif name == "dream1k":
        max_new_tokens = 2048
    elif f"max_tokens/{name}" in registry.list():
        max_new_tokens = registry.make(f"max_tokens/{name}")
    elif name == "tomato" or "temporal_bench" in name:
        max_new_tokens = 32
    elif name in [
        "mantis_eval",
        "mantis_eval_multi_choice",
        "mantis_eval_short_answer",
    ]:
        max_new_tokens = 32
    elif name in [
        "mantis_eval_legacy",
        "mantis_eval_legacy_multi_choice",
        "mantis_eval_legacy_short_answer",
    ]:
        max_new_tokens = 32
    elif name in ["mmsi_bench", "mmsi_bench_legacy_short_answer"]:
        max_new_tokens = 32
    elif name in ["mmiu", "mmiu_legacy_short_answer"]:
        max_new_tokens = 32
    elif name in ["muir_bench_legacy_answer_first", "muir_bench_legacy_answer_last"]:
        max_new_tokens = 448
    elif name in ["mmsi_bench_legacy_answer_first", "mmsi_bench_legacy_answer_last"]:
        max_new_tokens = 448
    elif name in ["mmiu_legacy_answer_first", "mmiu_legacy_answer_last"]:
        max_new_tokens = 448
    elif name.startswith("qv_highlights"):
        max_new_tokens = 40
    elif name == "mmmu_test":
        max_new_tokens = 64
    elif name in [
        "science_qa_img",
        "ai2_diagram_v2_mix_transparent",
        "a_okvqa_mc",
        "math_vista_v2",
    ]:
        max_new_tokens = 32
    elif name in [
        "vixmo_points_count",
        "vixmo_points_count_clip_63s",
        "academic_points_count_clip_63s",
        "lvvis_count_clip_63s",
        "burst_count_clip_63s",
        "vixmo_points_point_eval"
    ]:
        max_new_tokens = 2048
    elif "track" in name:
        max_new_tokens = 4096
    else:
        max_new_tokens = 12
    return max_new_tokens


def get_evaluation(
    name,
    seq_len,
    max_examples,
    for_inference=True,
    num_workers=2,
    device_batch_size=None,
    persistent_workers=False,
    include_image=False,
    num_wandb_examples=32,
    response_logits_only=False,
    reduce_loss_metrics_manually=False,
) -> InfDatasetEvaluatorConfig:
    """Gets the default evaluation config for task (or task:split string) `name`"""
    if ":" in name:
        name, split = name.split(":")
    else:
        split = None

    if name == "chart_qa_weighted":
        name = "chart_qa"
    if name == "coco_2014_vqa_multi":
        name = "coco_2014_vqa"

    eval_only_tasks = [
        "mmmu",
        "mme",
        "math_vista",
        "real_world_qa",
        "seed_bench",
        "mmbench",
        "sugar_crepe",
    ]
    eval_only_tasks += [task_name + "_test" for task_name in eval_only_tasks]
    if name == "tall_qa_count":
        task_name = "tally_qa"
    elif name in eval_only_tasks:
        task_name = name + "_test" if not name.endswith("_test") else name
    else:
        task_name = name
    test_eval_tasks = [
        "mme_test",
        "real_world_qa_test",
        "real_world_qa_test",
        "count_bench",
        "seed_bench_test",
        "sugar_crepe_test",
        "count_bench_from_caption",
        "pointing_test",
        "video_eval_mc",
        "nextqa_mc",
        "ego_schema_test",
        "perception_test_test",
        "mlvu_mc_test",
        "long_video_bench_w_subtitle_test",
        "motionbench_test",
    ]
    if split is None:
        split = "test" if task_name in test_eval_tasks else "validation"

    ds = DataLoaderConfig(
        dataset=task_name,
        sequence_length=seq_len,
        split=split,
        shuffle=True,
        drop_last=max_examples is not None and max_examples >= 0,
        num_workers=num_workers,
        pad="to_max",
        pin_memory=True,
        seed=691203,
        persistent_workers=persistent_workers,
    )

    if for_inference:
        evaluator = get_evaluator(name)
        evaluator.num_wandb_examples = num_wandb_examples
        evaluator.n_to_log = 0
        evaluator.save_predictions = None

        max_new_tokens = get_default_max_tokens(name)

        return InfDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            max_new_tokens=max_new_tokens,
            evaluator=evaluator,
            label="ai2_diagram" if "ai2_diagram" in name else name,
            data=ds,
            console_log_interval="${console_log_interval}",  # Use log interval in top-level config
            include_image=include_image,
        )

    else:
        return LossDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            label="ai2_diagram" if "ai2_diagram" in name else name,
            data=ds,
            console_log_interval="${console_log_interval}",  # Use log interval in top-level config
            response_logits_only=response_logits_only,
            reduce_loss_metrics_manually=reduce_loss_metrics_manually,
        )
