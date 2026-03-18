import argparse
import dataclasses
from dataclasses import asdict
from os.path import join
from typing import List
import numpy as np
from omegaconf import OmegaConf, omegaconf

from olmo.data.data_loader import WeightedDataset, KwargsMixture, DataLoaderConfig
from olmo.data.dynamic_packer import PackingConfig
from olmo.eval.eval_utils import get_evaluation
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.text_preprocessor import MessageWeight
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer
from olmo.train.trainer_config import FSDPConfig, BatchDivisor, SpeedMonitorConfig, TrainConfig, \
    WandbConfig, CompilerConfig
from olmo.util import prepare_torchrun_environment, select_checkpoint, clean_opt

IMAGE_ACADEMIC_DATASETS = [
    # Supervised datasets we want eval on
    "coco_2014_vqa_multi",
    "text_vqa",
    "okvqa",
    "chart_qa_weighted",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "a_okvqa_mc",
    "a_okvqa_da",

    # Some other datasets we might want to eval on
    "science_qa_img",
    "tabwmp_da",
    "st_qa",
    "tally_qa",

    # Multi-image
    "mantis_instruct_llava_665k_multi_multi_only",
    "mantis_instruct_nlvr2_multi_only",
    "mantis_instruct_spot-the-diff_multi_only",

    # # Other synthetic data, also downsampled since they are huge
    ("dv_qa", 10000),
    ("figure_qa", 10000),
    ("plot_qa", 20000),

    # Synthetic cosyn documents
    "cosyn_chart_exp",
    "cosyn_chemical_exp",
    # "cosyn_circuit_exp", # quality not good
    "cosyn_diagram_exp",
    "cosyn_document",
    # "cosyn_graphic_exp", # quality not good
    "cosyn_math_exp",
    "cosyn_music_exp",
    # "cosyn_nutrition_exp", # zero-shot evaluation dataset
    "cosyn_table_exp",

    # Synthetic cosyn multi-doc
    "cosyn_multidoc_chart_exp",
    "cosyn_multidoc_chemical_exp",
    "cosyn_multidoc_diagram_exp",
    "cosyn_multidoc_doc_exp",
    "cosyn_multidoc_music_exp",
    "cosyn_multidoc_table_exp",
]


VIDEO_ACADEMIC_DATASETS = [
    # academic qa
    ("tgif", 60_000),
    ("tvqa_with_sub", 60_000),
    ("paxion", 60_000),
    "llava_video_mc_academic",
    "llava_video_oe_academic",
    "perception_test",
    "nextqa_mc",
    "news_video_qa_filtered",
    "how2qa",
    'sutd_trafficqa',
    'social_iq2',
    "sportsqa_oe",
    "cinepile_with_sub",
    "clevrer",
    "funqa",
    "star",
    "intent_qa",
    "video_localized_narratives",
    "road_text_vqa",
    "countix_oe",
    "camerabench_qa",
    "motionbench_train",

    # academic other
    # ("ssv2_qa", 60_000),
    ("moments_in_time_qa", 60_000),
    ("kinetics_qa", 60_000),
    "charades_sta_all_qa",
    "coin_all_qa",
    "youcook2_all_qa",
    "activitynet_all_qa",
    "ego4d_all",
    "epic_kitchens_qa",
    "video_localized_narratives_caption",
    "qv_highlights",

    # No captions since we put this in the `demo` group

    # our QA
    "vixmo_syn_video_capqa_v2",  # 200K general qa
    "vixmo_syn_video_capqa_with_sub_v2",  # 100K subtitle qa
]


TRACKING_MIXTURE = [
    ["track", [
        "mevis_track",
        "ref_yt_vos_track",
        "lv_vis_track",
        ("vicas_track", 30000),
        "revos_track",
        "molmo2_video_track",
    ], 0.4],
    ["track_dist_tail", [
        "burst_track",
        "ref_davis17_track",
        "yt_vis_track",
        "moca_track",
    ], 0.05],
    ["ground", [
        "mevis_ground",
        "lv_vis_ground",
        ("vicas_ground", 30000),
        "revos_ground",
        "molmo2_video_track_ground",
    ], 0.2],
    ["ground_dist_tail", [
        "burst_ground",
        "moca_ground",
    ], 0.01],
    ["single_point_track", [
        "mevis_single_point_track",
        "lv_vis_single_point_track",
        ("vicas_single_point_track", 30000),
        "revos_single_point_track",
        "molmo2_video_single_point_track",
    ], 0.15],
    ["sot_from_bbox", [
        "webuav_single_point_track",
        "got10k_single_point_track",
        "vasttrack_single_point_track",
        "trackingnet_single_point_track",
    ], 0.2],
    ["sot_from_bbox_dist_tail", [
        "burst_single_point_track",
        "lvosv1_single_point_track",
        "lvosv2_single_point_track",
        "lasot_single_point_track",
        "uwcot_single_point_track",
        "webuot_single_point_track",
        "latot_single_point_track",
        "tnl2k_single_point_track",
        "tnllt_single_point_track",
    ], 0.1],
]


# Updated tracking mixture used in MolmoPoint
TRACKING_MIXTURE_v2_1 = [
    ["track", [
        "mevis_track",
        "ref_yt_vos_track",
        "lv_vis_track",
        ("vicas_track", 30000),
        "revos_track",
        "molmo2_video_track",
        "molmo_point_track_any",
        ("molmo_point_track_syn", 10000),
    ], 0.45],
    ["track_dist_tail", [
        "burst_track",
        "ref_davis17_track",
        "yt_vis_track",
        "moca_track",
    ], 0.07],
    ["ground", [
        "mevis_ground",
        "lv_vis_ground",
        ("vicas_ground", 30000),
        "revos_ground",
        "molmo2_video_track_ground",
    ], 0.2],
    ["ground_dist_tail", [
        "burst_ground",
        "moca_ground",
    ], 0.01],
    ["single_point_track", [
        "mevis_single_point_track",
        "lv_vis_single_point_track",
        ("vicas_single_point_track", 30000),
        "revos_single_point_track",
        "molmo2_video_single_point_track",
    ], 0.15],
    ["sot_from_bbox", [
        "webuav_single_point_track",
        "got10k_single_point_track",
        "vasttrack_single_point_track",
        "trackingnet_single_point_track",
    ], 0.2],
    ["sot_from_bbox_dist_tail", [
        "burst_single_point_track",
        "lvosv1_single_point_track",
        "lvosv2_single_point_track",
        "lasot_single_point_track",
        "uwcot_single_point_track",
        "webuot_single_point_track",
        "latot_single_point_track",
        "tnl2k_single_point_track",
        "tnllt_single_point_track",
    ], 0.1],
]


def get_model(checkpoint, model):
    model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
    video_preprocessor_cfg = VideoPreprocessorConfig(
        pooling_h=3,
        pooling_w=3,
        time_mode="per-frame-compact",
        max_frames=128,
        time_sampling=True,
        loading_method="torchcodec_exact",
        frame_sample_mode="uniform_last_frame",
        max_fps=[2],
    )
    if isinstance(model_cfg.mm_preprocessor, MultiCropConfig):
        # Might be starting from a `Molmo` not a Molmo2` model
        kwargs = {field.name: getattr(model_cfg.mm_preprocessor, field.name) for field in dataclasses.fields(MultiCropConfig)}
        image_preprocessor_cfg = MultiCropConfig(**kwargs)
    else:
        image_preprocessor_cfg = model_cfg.mm_preprocessor.image

    model_cfg = Molmo2Config(
        llm=model_cfg.llm,
        vision_backbone=model_cfg.vision_backbone,
        data_formatter=model_cfg.data_formatter,
        mm_preprocessor=Molmo2PreprocessorConfig(
            video=video_preprocessor_cfg,
            image=image_preprocessor_cfg,
        ),
        bi_directional_attn=model_cfg.bi_directional_attn
    )

    # Fine-tuning settings
    model_cfg.vision_backbone.pooling_attention_mask = True
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.mm_preprocessor.video.max_subtitle_tokens = None
    model_cfg.data_formatter.p_multi_point_all_image = 0.5
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0

    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model_v2"
    model_cfg.data_formatter.message_format = "qwen3"
    model_cfg.data_formatter.system_prompt = "demo_or_style_v2"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"

    # Multi-image settings
    model_cfg.mm_preprocessor.image.max_multi_image_crops = 8
    model_cfg.mm_preprocessor.image.max_images = 5

    # Good enough for 128 frames
    model_cfg.llm.max_sequence_length = 4096*4

    # Reduce shared memory requirements
    model_cfg.vision_backbone.normalize_on_gpu = True

    return model_cfg


def get_training_mixture(name):
    if name == "debug":
        hardcode_weight = 0.001
        training_mixture = [
            ["demo", ["pixmo_ask_model_anything"], 0.10],
            ["video_academic", ["llava_video_mc_academic"], 0.2],
            ["image_academic", ["chart_qa_weighted"], 0.25],
            ["tracking", ["mevis_track"], 0.10],
            ["nlp", ["tulu4"], 0.1 - hardcode_weight],
            ["hardcodes", ["molmo2_hardcodes"], hardcode_weight]
        ]
    elif name == "tracking":
        training_mixture = TRACKING_MIXTURE
    elif name == "molmo2":
        pointing_high_res = 0.30
        point_weight = MessageWeight(weight=0.2, root_length=False, root_subsegments=False)
        cap_weight = MessageWeight(weight=0.1, root_length=False, root_subsegments=False)
        if "no-weight" in name:
            cap_weight, point_weight = None, None
        video_pointing = [
            ["vixmo_points_minmax_0_5", 0.4],
            ["vixmo_points_minmax_6_25", 0.3],
            ["vixmo_points_minmax_26_60", 0.1],
            ["academic_points_clip_63s_2fps", 0.2]
        ]
        hardcode_weight = 0.0005
        academic = IMAGE_ACADEMIC_DATASETS + VIDEO_ACADEMIC_DATASETS
        max_text_len = None
        track_mix = TRACKING_MIXTURE
        total = np.sum([x[-1] for x in track_mix])
        track_tasks = []
        for name, datasets, weight in track_mix:
            for dataset in datasets:
                if isinstance(dataset, tuple):
                    dataset, root_size_factor = dataset
                else:
                    root_size_factor = None
                track_tasks.append(WeightedDataset(
                    dataset, sampling_rate=float(weight/total), root_size_factor=root_size_factor,
                    message_weight=point_weight
                ))

        video_pointing_tasks = []
        for task, weight in video_pointing:
            video_pointing_tasks.append(WeightedDataset(
                task, sampling_rate=weight, root_size_factor=1,
                message_weight=point_weight
            ))

        training_mixture = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 100000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_multi_image_qa_multi_only_max5",
                "molmo2_human_qa",
                WeightedDataset("vixmo3_top_level_captions_min_3", root_size_factor=None,
                                sampling_rate=1.5, message_weight=cap_weight),
            ], 0.15],
            ["video_academic", VIDEO_ACADEMIC_DATASETS, 0.2],
            ["image_academic", IMAGE_ACADEMIC_DATASETS, 0.25],
            ["tracking", track_tasks, 0.15],
            ["video_pointing", video_pointing_tasks, 0.15],
            ["image_pointing", [
                WeightedDataset("pixmo_multi_points", root_size_factor=200000,
                                message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("pixmo_points_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("pixmo_count_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("pixmo_points_high_freq_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("cosyn_point", message_weight=point_weight, override_p_high_res=pointing_high_res),
            ], 0.1],
            ["nlp", ["tulu4"], 0.1 - hardcode_weight],
            ["hardcodes", ["molmo2_hardcodes"], hardcode_weight]
        ]
    elif name in ["molmo_point", "molmo_point_long_context"]:
        pointing_high_res = 0.30
        point_weight = MessageWeight(weight=0.2, root_length=False, root_subsegments=False)
        cap_weight = MessageWeight(weight=0.1, root_length=False, root_subsegments=False)
        if "no-weight" in name:
            cap_weight, point_weight = None, None
        if name == "molmo_point_long_context":
            video_pointing = [
                ["vixmo_points_oversample_no_clip", 0.8],
                ["academic_points_clip_63s_2fps", 0.2]
            ]
        else:
            video_pointing = [
                ["vixmo_points_oversample", 0.8],
                ["academic_points_clip_63s_2fps", 0.2]
            ]
        hardcode_weight = 0.0005
        academic = IMAGE_ACADEMIC_DATASETS + VIDEO_ACADEMIC_DATASETS
        max_text_len = None
        track_mix = TRACKING_MIXTURE_v2_1
        total = np.sum([x[-1] for x in track_mix])
        track_tasks = []
        for name, datasets, weight in track_mix:
            for dataset in datasets:
                if isinstance(dataset, tuple):
                    dataset, root_size_factor = dataset
                else:
                    root_size_factor = None
                track_tasks.append(WeightedDataset(
                    dataset, sampling_rate=float(weight/total), root_size_factor=root_size_factor,
                    message_weight=point_weight
                ))

        video_pointing_tasks = []
        for task, weight in video_pointing:
            video_pointing_tasks.append(WeightedDataset(
                task, sampling_rate=weight, root_size_factor=1,
                message_weight=point_weight
            ))

        training_mixture = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 100000),
                "pixmo_cap_qa_as_user_qa",
                "correction_qa_multi_only_max5",
                "vixmo_human_qa",
                WeightedDataset("vixmo3_top_level_captions_min_3", root_size_factor=None,
                                sampling_rate=1.5, message_weight=cap_weight),
            ], 0.15],
            ["video_academic", VIDEO_ACADEMIC_DATASETS, 0.2],
            ["image_academic", IMAGE_ACADEMIC_DATASETS, 0.25],
            ["tracking", track_tasks, 0.12],
            ["video_pointing", video_pointing_tasks, 0.11],
            ["image_pointing", [
                WeightedDataset("pixmo_multi_points", root_size_factor=200000,
                                message_weight=point_weight),
                WeightedDataset("pixmo_points_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("pixmo_count_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("pixmo_points_high_freq_train", message_weight=point_weight, override_p_high_res=pointing_high_res),
                WeightedDataset("cosyn_point", message_weight=point_weight, override_p_high_res=pointing_high_res),
            ], 0.07],
            ["nlp", ["tulu4"], 0.1 - hardcode_weight],
            ["hardcodes", ["molmo2_hardcodes"], hardcode_weight]
        ]
    else:
        raise NotImplementedError(name)
    root_size_mixture: List[KwargsMixture] = []
    for name, submixture, rate in training_mixture:
        submixture = [WeightedDataset.build(x) for x in submixture]
        root_size_mixture.append(KwargsMixture(rate, submixture, name))
    return root_size_mixture


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("mixture", default="0.0.1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", default="video")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--device_batch_size", default=2, type=int)
    parser.add_argument("--max_loss_examples", default=2048, type=int)
    parser.add_argument("--max_inf_eval_examples", default=1280, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--cp_degree", default=1, type=int)
    args, other_args = parser.parse_known_args()

    if args.mixture == "debug":
        loss_eval_tasks = ["llava_video_oe_academic", "pixmo_ask_model_anything"]
        eval_tasks = ["chart_qa", "mvbench", "mevis_track_eval_1fps:test"]
    elif args.mixture.startswith("pointing"):
        loss_eval_tasks = []
        eval_tasks = ["pointing_eval_v2:test", "vixmo_points_count:val", "pixmo_count_counting:validation"]
    elif args.mixture.startswith("vpointing"):
        loss_eval_tasks = []
        eval_tasks = ["vixmo_points_count:val"]
    elif args.mixture.startswith("image-only-v5"):
        loss_eval_tasks = []
        eval_tasks = [
            "chart_qa", "info_qa", "coco_2014_vqa_multi", "pointing_eval_v2:test"
        ]
    else:
        loss_eval_tasks = ["llava_video_oe_academic", "pixmo_ask_model_anything", "pixmo_cap"]
        eval_tasks = [
            "chart_qa", "info_qa", "coco_2014_vqa_multi",
            "pixmo_clocks",
            "pointing_eval_v2:test",
            "muir_bench:test",
            "mvbench",
            "vixmo_points_count:val"
        ]

    training_mixture = get_training_mixture(args.mixture)
    seq_len = args.seq_len

    checkpoint = select_checkpoint(args.checkpoint)
    model_cfg = get_model(checkpoint, args.model)

    if args.debug:
        checkpoint = None

        # Use a dummy model, but one still based on the input checkpoint
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 1
        if hasattr(model_cfg, "vision_backbone"):
            vit = model_cfg.vision_backbone.vit
            model_cfg.vision_backbone.vit_layers = [-1, -2]
        else:
            model_cfg.image_layers = [0]
            model_cfg.connector.vit_layers = [-1, -2]
            vit = model_cfg.vit
        vit.init_path = None
        vit.image_num_layers = 2
        args.num_workers = 2
        args.prefetch_factor = 2

    num_workers = args.num_workers
    evaluations = []
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            None,
            device_batch_size=args.device_batch_size*2,
            max_examples=args.max_inf_eval_examples,
            num_workers=num_workers,
        )
        evaluation.data.pad = None
        evaluation.data.max_text_seq_len = 128  # Only needs to be enough for the question
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        evaluations.append(evaluation)

    loss_evaluations = []
    for task in loss_eval_tasks:
        evaluation = get_evaluation(
            task,
            seq_len=seq_len,
            for_inference=False,
            device_batch_size=args.device_batch_size*2,
            max_examples=args.max_loss_examples,
            num_workers=num_workers,
        )
        evaluation.data.max_text_seq_len = None
        evaluation.data.pad = "to_max"
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        loss_evaluations.append(evaluation)

    log_interval = 1 if args.debug else 20
    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder=omegaconf.MISSING,
        seed=6198,
        dry_run=False,

        wandb=None if args.debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval,
            allow_resume=False,
            finish_on_sigterm=True
        ),
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=True,
        data=DataLoaderConfig(
            kwargs_mixture=training_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            max_text_seq_len=None,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
            packing=PackingConfig(buffer_size=48, image_weight=30, shortcut_max_len_images=False,
                                  cp_world_size=args.cp_degree)
        ),
        ft_connector=True,
        ft_llm=not args.debug,
        ft_vit=not args.debug,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            frame_selector_learning_rate=1e-4,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            frame_selector_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(fsdp2=True),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=2000,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=get_world_size() if args.debug else 128,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=300000,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        inf_evaluators=evaluations,
        evaluators=loss_evaluations,
        inf_eval_interval=-1,
        eval_interval=-1,
        save_final_unsharded_checkpoint=False,
        save_final_optim=False,
        response_logits_only=True,
    )

    cfg.parallelism.context_parallel_config.degree = args.cp_degree

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    conf = OmegaConf.to_object(conf)

    if conf.parallelism.context_parallel_config.degree > 1:
        conf.model.cp_enabled = True
        conf.compile = None  # compilation is not supported with context parallelism

    run_trainer(conf)


if __name__ == '__main__':
    main()