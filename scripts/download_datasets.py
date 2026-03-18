#!/usr/bin/env python3
"""CLI tool to download datasets training/eval data"""
import argparse
import logging
import sys
from typing import List

from launch_scripts.sft import VIDEO_ACADEMIC_DATASETS, IMAGE_ACADEMIC_DATASETS
from olmo.data.get_dataset import get_all_dataset_classes, get_dataset_class_by_name
from olmo.util import prepare_cli_environment


def download_datasets(datasets: List, n_procs: int = 8):
    failed_datasets = []
    for i, dataset in enumerate(datasets, 1):
        name = dataset.__name__
        logging.info(f"[{i}/{len(datasets)}] Downloading dataset: {name}")
        try:
            dataset.download(n_procs=n_procs)
            logging.info(f"✓ Successfully downloaded: {name}")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logging.error(f"✗ Failed to download {name}: {e}")
            failed_datasets.append((name, str(e)))

    # Summary
    logging.info("\n" + "="*60)
    logging.info(f"Download complete: {len(datasets) - len(failed_datasets)}/{len(datasets)} succeeded")

    if failed_datasets:
        logging.error("\nFailed datasets:")
        for name, error in failed_datasets:
            logging.error(f"  - {name}: {error}")
        return 1
    return 0

DATASET_GROUPS = {
    "image_academic": IMAGE_ACADEMIC_DATASETS,
    "video_academic": VIDEO_ACADEMIC_DATASETS,
    "pixmo": [
        "pixmo_cap",
        "pixmo_multi_points",
        "pixmo_points_train",
        "pixmo_count_train",
        "cosyn_point"
    ],
    "image_pointing": [
        "pixmo_multi_points",
        "pixmo_points_train",
        "pixmo_count_train",
        "cosyn_point"
    ],
    "video_pointing": [
        "vixmo_points_oversample"
        "academic_points_clip_63s_2fps"
    ],
    "video_tracking": [

        # mot
        "mevis_track",
        "ref_yt_vos_track",
        "lv_vis_track",
        "vicas_track",
        "revos_track",
        "burst_track",
        "ref_davis17_track",
        "yt_vis_track",
        "moca_track",

        "molmo2_video_track",
        "molmo_point_track_any",
        "molmo_point_track_syn",

        # sot
        "webuav_single_point_track",
        "got10k_single_point_track",
        "vasttrack_single_point_track",
        "trackingnet_single_point_track",
        "lvosv1_single_point_track",
        "lvosv2_single_point_track",
        "lasot_single_point_track",
        "uwcot_single_point_track",
        "webuot_single_point_track",
        "latot_single_point_track",
        "tnl2k_single_point_track",
        "tnllt_single_point_track",
    ],
    "demo": [
        "pixmo_ask_model_anything",
        "pixmo_cap",
        "pixmo_cap_qa_as_user_qa",
        "pixmo_multi_image_qa",
        "vixmo_human_qa",
        "vixmo3_top_level_captions_min_3"
    ],
}


def main():
    prepare_cli_environment()

    parser = argparse.ArgumentParser(
        description="Download datasets for Molmo training/evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  all                    - Download all datasets
  Individual datasets:   - See ALL_DATASETS list in script

Examples:
  # Download a single dataset
  python {sys.argv[0]} text_vqa

  # Download multiple datasets
  python {sys.argv[0]} text_vqa doc_qa chart_qa

  # Download all datasets
  python {sys.argv[0]} all

  # Download dataset group
  python {sys.argv[0]} video_tracking

  # Download with more parallel processes
  python {sys.argv[0]} text_vqa --n-procs 16
"""
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset name(s) to download, or 'all' for all datasets"
    )

    parser.add_argument(
        "--n-procs",
        type=int,
        default=8,
        help="Number of parallel processes to use for downloading (default: 8)"
    )

    args = parser.parse_args()
    datasets_to_download = {}  # dictionary to preserve insertion order
    dataset_classes = get_all_dataset_classes()
    for name in args.datasets:
        if name == "all":
            datasets_to_download.update({k: None for k in sorted(dataset_classes, key=lambda cls: cls.__name__)})
        elif name in DATASET_GROUPS:
            for dataset_name in DATASET_GROUPS[name]:
                if isinstance(dataset_name, tuple):
                    dataset_name = dataset_name[0]
                datasets_to_download[get_dataset_class_by_name(dataset_name)] = None
        else:
            matches = [cls for cls in dataset_classes if cls.__name__.lower() == name.lower()]
            if len(matches) == 1:
                datasets_to_download[matches[0]] = None
            else:
                try:
                    datasets_to_download[get_dataset_class_by_name(name)] = None
                except ValueError:
                    raise ValueError(f"No dataset or dataset class found with name {name}.")

    return download_datasets(list(datasets_to_download.keys()), n_procs=args.n_procs)


if __name__ == "__main__":
    sys.exit(main())
