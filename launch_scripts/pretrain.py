import argparse
import logging
from dataclasses import replace
from typing import cast

from omegaconf import omegaconf, OmegaConf

from olmo.data.dynamic_packer import PackingConfig
from olmo.eval.eval_utils import get_evaluation
from olmo.models.molmo2.molmo2 import Molmo2, Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.models.molmo_point.molmo_point import MolmoPointConfig, MolmoPointPreprocessorConfig
from olmo.models.molmo_point.molmo_point_connector import ConnectorConfig
from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter
from olmo.preprocessing.data_formatter import DataFormatter
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.data.pixmo_datasets import PixMoCap
from olmo.model_configs import DEBUG_MOLMO, VISION_BACKBONES, LLMS, DEBUG_MOLMO2
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.model import FSDPWrapStrategy
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig, ImagePaddingEmbed
from olmo.train.run_trainer import run_trainer

from olmo.data.data_loader import DataLoaderConfig, KwargsMixture, WeightedDataset
from olmo.train.trainer_config import BatchDivisor, SpeedMonitorConfig, \
    CompilerConfig, TrainConfig, WandbConfig, FSDPConfig, FSDPPrecision
from olmo.util import clean_opt, prepare_torchrun_environment


log = logging.getLogger("train")


if __name__ == "__main__":
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a captioner")
    parser.add_argument("llm", choices=["debug"] + list(LLMS.keys()))
    parser.add_argument("--model", default="molmo2", choices=["debug", "molmo", "molmo2", "molmo_point"])
    parser.add_argument("--vision_backbone", choices=list(VISION_BACKBONES.keys()), default="siglip2")
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--n_eval_examples", default=2048, type=int)
    parser.add_argument("--device_eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation per device")
    parser.add_argument("--seq_len", default=2536, type=int,
                        help="Maximum sequence length to pad examples to.")
    parser.add_argument("--vit_layers", type=int, nargs="+", default=[-3, -9])
    parser.add_argument("--warmup_factor", type=int, default=1)
    parser.add_argument("--nlp", default=0.1, type=float,
                        help="Fraction of NLP data in the mixture")
    parser.add_argument("--pointing", default=0.3, type=float,
                        help="Fraction of pointing data in the mixture")
    args, other_args = parser.parse_known_args()

    # Setup the model config
    seq_len = args.seq_len
    debug = args.llm in ["debug"]
    if debug:
        model_cfg = DEBUG_MOLMO2
        model_cfg.data_formatter.system_prompt = 'style_and_length_v2'
        global_batch_size = 8
        model_init = None
        eval_interval = 20
        log_interval = 5
        eval_examples = 64
        duration = 200
    else:
        eval_examples = args.n_eval_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        n = len(PixMoCap("train", "captions"))
        duration = 4 * (n + global_batch_size - 1) // global_batch_size
        eval_interval = 1000
        # vit_layers = [-2, -9] if args.vision_backbone == "openai" else [-3, -9]
        vit_layers = args.vit_layers

        image_vit = VISION_BACKBONES[args.vision_backbone]
        if args.model == "molmo":
            model_cfg = MolmoConfig(
                llm=replace(
                    LLMS[args.llm],
                    residual_dropout=0.0,
                    response_residual_dropout=0.1,
                    additional_vocab_size=128,
                ),
                vision_backbone=MolmoVisionBackboneConfig(
                    vit=VISION_BACKBONES[args.vision_backbone],
                    vit_layers=vit_layers,
                    image_padding_embed=ImagePaddingEmbed.pad_and_partial_pad if args.vision_backbone == "openai" else None,
                    pooling_attention_mask=True
                ),
                data_formatter=DataFormatter(
                    system_prompt='style_and_length_v2',
                    message_format="qwen3",
                    pointing_format="html-v1",
                    always_start_with_space=False,
                ),
                mm_preprocessor=MolmoPreprocessorConfig(
                    use_single_crop_col_tokens=False,
                    use_single_crop_start_token=True,
                    crop_mode="overlap-and-resize-c2",
                    max_crops=8 if args.vision_backbone in ["siglip", "siglip2"] else 12,
                    overlap_margins=(4, 4)
                )
            )
        elif args.model == "molmo2":
            model_cfg = Molmo2Config(
                llm=replace(
                    LLMS[args.llm],
                    residual_dropout=0.0,
                    response_residual_dropout=0.1,
                    additional_vocab_size=128,
                ),
                vision_backbone=MolmoVisionBackboneConfig(
                    vit=VISION_BACKBONES[args.vision_backbone],
                    vit_layers=vit_layers,
                    image_padding_embed=ImagePaddingEmbed.pad_and_partial_pad if args.vision_backbone == "openai" else None,
                    pooling_attention_mask=True
                ),
                data_formatter=DataFormatter(
                    system_prompt='style_and_length_v2',
                    message_format="qwen3",
                    pointing_format="html-v1",
                    always_start_with_space=False,
                ),
                mm_preprocessor=Molmo2PreprocessorConfig(
                    # Max frames 16 means the model will take 16 crops as input, this is more than
                    # needed for single images but significantly improves packing efficiency
                    video=VideoPreprocessorConfig(max_frames=16),
                    image=MultiCropConfig(
                        use_single_crop_col_tokens=False,
                        use_single_crop_start_token=True,
                        crop_mode="overlap-and-resize-c2",
                        max_crops=8 if args.vision_backbone in ["siglip", "siglip2"] else 12,
                        overlap_margins=(4, 4)
                    )
                )
            )
        if args.model == "molmo_point":
            # MolmoPoint was pre-trained with fewer steps but up to 16 images per a sequence
            # to improve packing
            duration = 23000
            model_cfg = MolmoPointConfig(
                data_formatter=MolmoPointDataFormatter(
                    system_prompt='style_and_length_v2',
                    message_format="qwen3",
                    include_point_number="no_space_id_last",
                ),
                llm=replace(
                    LLMS[args.llm],
                    residual_dropout=0.0,
                    response_residual_dropout=0.1,
                    additional_vocab_size=128,
                    can_predict_extra_tokens=True
                ),
                patch_embed_dim=512,
                vit=VISION_BACKBONES[args.vision_backbone],
                connector=ConnectorConfig(
                    vit_layers=vit_layers,
                    image_projector="mlp",
                    positional_embeddings=None,
                    query="mean",
                    pooling_out_layer=False,
                    normalize_on_gpu=True
                ),
                mm_preprocessor=MolmoPointPreprocessorConfig(
                    video=VideoPreprocessorConfig(
                        max_frames=16,
                        pooling_h=3,
                        pooling_w=3,
                        frame_sample_mode="uniform_last_frame",
                        time_sampling=True,
                        loading_method="torchcodec_exact",
                        max_fps=[2],
                        per_frame_special_token=True,
                        use_frame_special_tokens=True
                    ),
                    image=MultiCropConfig(
                        use_single_crop_start_token=True,
                        use_single_crop_col_tokens=False,
                        use_col_tokens=True,
                        crop_mode="overlap-and-resize-c2",
                        max_crops=8,
                        max_images=1,
                        max_multi_image_crops=1
                    )),
                no_more_points_class=True,
                layer_norm_x=True,
                norm_logits=True,
                patch_embedding_kind="image_feature0",
                embed_selected_vit_patch="linear",
                patch_location="3x3",
                bi_directional_attn="image_tokens",
            )
        else:
            raise NotImplementedError(args.model)

    # Evaluator for the val loss on the captioning data
    evaluator = LossDatasetEvaluatorConfig(
        label="val",
        max_examples=eval_examples,
        device_batch_size=args.device_eval_batch_size,
        console_log_interval="${console_log_interval}",
        data=DataLoaderConfig(
            seed="${seed}",
            dataset="pixmo_cap_with_transcripts",
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    # Setup the dataset mixture
    dataset, mixture, kwargs_mixture = None, None, None
    inf_evaluators = []
    if args.pointing:
        kwargs_mixture = [
            KwargsMixture(1.0 - args.pointing - args.nlp, [WeightedDataset("pixmo_cap_with_transcripts")]),
            KwargsMixture(args.pointing, [
                WeightedDataset("pixmo_points_train"),
                WeightedDataset("pixmo_count_train"),
                WeightedDataset("pixmo_points_high_freq_train"),
                WeightedDataset("cosyn_point"),
            ]),
        ]
        if args.nlp:
            assert seq_len > 2304
            duration = 31000
            kwargs_mixture.append(KwargsMixture(args.nlp, [WeightedDataset("tulu4_max_2304")]))

        for task in ["point_bench:test", "pixmo_count_counting:validation"]:
            evaluation = get_evaluation(
                task,
                None,
                device_batch_size=4,
                max_examples=args.n_eval_examples,
                num_workers=2,
            )
            evaluation.data.pad = None
            evaluation.data.max_text_seq_len = 196  # Only needs to be enough for the question
            evaluation.data.persistent_workers = True
            evaluation.data.prefetch_factor = 4
            inf_evaluators.append(evaluation)
    elif args.nlp:
        # Most NLP data will get packed into the captions, but increase a little bit to
        # compensate for the ones that are not
        duration = int(duration * 1.01)
        assert seq_len > 2304
        mixture = {
            args.dataset: 1-args.nlp,
            "tulu4_max_2304": args.nlp
        }
    else:
        dataset = args.dataset

    # Put together into the full trainer config
    cfg = TrainConfig(
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        compile_loss=True,
        model=model_cfg,
        data=DataLoaderConfig(
            dataset=dataset,
            mixture=mixture,
            kwargs_mixture=kwargs_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
            packing=PackingConfig(48, shortcut_max_len_images=False) if args.nlp else None
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=2e-4,
            vit_learning_rate=6e-6,
            llm_learning_rate=2e-5,
            frame_selector_learning_rate=1e-4,
            metrics_log_interval=-1
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200 // args.warmup_factor,
            vit_t_warmup=2000 // args.warmup_factor,
            llm_t_warmup=2000 // args.warmup_factor,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=None,
        save_overwrite=debug,
        save_interval=4000,
        allow_resume=True,
        save_num_checkpoints_to_keep=1,
        save_final_unsharded_checkpoint=False,
        global_train_batch_size=global_batch_size,
        device_train_microbatch_size=4,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=eval_interval,
        inf_eval_interval=2000,
        response_logits_only=True,
        inf_evaluators=inf_evaluators,
        evaluators=[
            evaluator,
            replace(evaluator, data=replace(evaluator.data, dataset="pixmo_cap"), label="caption_val")
        ]
    )

    # Update the trainer config w/CLI args and then run
    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    run_trainer(cfg)



