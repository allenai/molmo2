import argparse
import logging
from os.path import join

from omegaconf import omegaconf, OmegaConf

from olmo.data.data_loader import DataLoaderConfig
from olmo.data.dynamic_packer import PackingConfig
from olmo.eval.eval_utils import get_evaluation
from olmo.models.model_config import BaseModelConfig
from olmo.models.molmo_point.molmo_point import MolmoPointConfig
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer as train
from olmo.train.trainer_config import TrainConfig, WandbConfig, FSDPConfig, CompilerConfig, \
    BatchDivisor, SpeedMonitorConfig, InfEvalConfig
from olmo.util import clean_opt, select_checkpoint, prepare_torchrun_environment

log = logging.getLogger("train")


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a GUI pointing model")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    args, other_args = parser.parse_known_args()

    args.checkpoint = select_checkpoint(args.checkpoint)
    model_init = args.checkpoint
    model_cfg = BaseModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")
    if isinstance(model_cfg, MolmoPointConfig):
        model_cfg.mm_preprocessor.image.max_crops = 48
        model_cfg.mm_preprocessor.image.max_images = 1
        model_cfg.mm_preprocessor.image.max_multi_image_crops = 8
        model_cfg.mm_preprocessor.video.max_frames = 49
    else:
        raise NotImplementedError()
    model_cfg.llm.max_sequence_length = 12288

    inf_evals = []
    for task in ["screen_spot_pro_click:test", "screen_spot_v2_click:test"]:
        evaluator = get_evaluation(task, None,
                                   num_workers=2,
                                   device_batch_size=2,
                                   max_examples=None,
                                   persistent_workers=True)
        evaluator.data.pad = None
        inf_evals.append(evaluator)

    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder=omegaconf.MISSING,
        seed=6198,
        initial_model_checkpoint=model_init,
        dry_run=False,
        wandb=WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=10
        ),
        model=model_cfg,
        data=DataLoaderConfig(
            dataset="molmo2_syn_point",
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=12288,
            max_text_seq_len=None,
            seed=95818,
            num_workers=2,
            packing=PackingConfig(32, image_weight=10, shortcut_max_len_images=True),
            pad="to_max",
            pin_memory=True,
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
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
        allow_resume=True,
        save_overwrite=True,
        load_path=None,
        compile=CompilerConfig(),
        fused_loss=False,
        save_interval=500,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=128,
        device_train_microbatch_size=4,
        time_limit=None,
        max_duration=2000,
        stop_at="${max_duration}",
        max_grad_norm=1,
        inf_eval_config=InfEvalConfig(False, True, True),
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=10,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        eval_interval=-1,
        inf_eval_interval=500,
        evaluators=[],
        response_logits_only=True,
        inf_evaluators=inf_evals,
    )
    conf = OmegaConf.create(cfg)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist([clean_opt(arg) for arg in other_args]))
    conf = OmegaConf.to_object(conf)
    train(conf)


if __name__ == '__main__':
    main()