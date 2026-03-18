
import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from os import environ
from os.path import join, exists, relpath, dirname
from pathlib import Path
from typing import List
import numpy as np

import datasets
from PIL import Image
from cached_path import cached_path
from huggingface_hub import hf_hub_download, snapshot_download

from olmo.data.dataset_builders.count_bench_qa import CountQaBuilder
from olmo.data.dataset_builders.tabe_wmpd import TabMwpBuilder
from olmo.io import read_file, write_json, list_directory

from olmo.util import flatten_list, resource_path

from olmo.data.dataset import Dataset, DatasetBase, DATA_HOME, Ai2HfDataset, HfDataset
from olmo.data.dataset_builders.ai2d import Ai2dDatasetBuilder
from olmo.data.dataset_builders.dv_qa import DvQaBuilder
from olmo.data.dataset_builders.figure_qa import FigureQaBuilder
from olmo.data.dataset_builders.plot_qa import PlotQaBuilder
from olmo.data.utils import save_local_dataset, maybe_download_and_unzip, save_images


log = logging.getLogger(__name__)


# FIXME all these class should use ACADEMIC_DATASETS
# Switch once we get a break between training runs
if DATA_HOME is not None:
    ACADEMIC_DATASETS = join(DATA_HOME, "academic_datasets")
else:
    ACADEMIC_DATASETS = None


COCO2014_URLS_IMAGES = {
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "test2015": "http://images.cocodataset.org/zips/test2015.zip",
}

COCO2017_URLS_IMAGES = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
}

VG_URLS = {
    "VG_100K_2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    "VG_100K": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
}


def _ensure_vg_downloaded():
    for name, url in VG_URLS.items():
        maybe_download_and_unzip(join(DATA_HOME, "images", "vg"), url, name)


def _ensure_coco2014_downloaded():
    for name, url in COCO2014_URLS_IMAGES.items():
        maybe_download_and_unzip(join(DATA_HOME, "images", "coco"), url, name)


def _ensure_coco2017_downloaded():
    for name, url in COCO2017_URLS_IMAGES.items():
        maybe_download_and_unzip(join(DATA_HOME, "images", "coco"), url, name)


class TextVqa(Ai2HfDataset):
    IMAGE_HOME = join(DATA_HOME, "text_vqa")
    HF_SOURCE = "allenai/molmo2-text-vqa"
    IMAGE_URLS = {
        "train_images": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        "test_images": "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
    }

    @classmethod
    def download(cls, n_procs=1):
        for name, url in cls.IMAGE_URLS.items():
            maybe_download_and_unzip(join(cls.IMAGE_HOME, name), url)
        for k in ["train", "validation", "test"]:
            datasets.load_dataset_builder(cls.HF_SOURCE, name=k).download_and_prepare()

    def __init__(self, split: str):
        super().__init__(split, self.HF_SOURCE, "text_vqa")


class OkVqa(Dataset):
    HF_SOURCE = "allenai/molmo2-okvqa"
    SPLITS = ["train", "val", "test"]

    @classmethod
    def download(cls, n_procs=1):
        _ensure_coco2014_downloaded()
        for k in ["train", "val"]:
            datasets.load_dataset_builder(cls.HF_SOURCE, name=k).download_and_prepare()

    def __init__(self, split: str, flatten_annotations=False):
        self.data = datasets.load_dataset(self.HF_SOURCE, split, keep_in_memory=flatten_annotations)["train"]
        self.flatten_annotations = flatten_annotations
        if flatten_annotations:
            self.data = flatten_list([dict(ex, qas=[q]) for q in ex["qas"]] for ex in self.data)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = dict(self.data[item])
        message_list = []
        for q in example["qas"]:
            message_list.append(dict(
                question=q["question"],
                answers=[x["raw_answer"] for x in q["answers"]],
                style="okvqa",
            ))
        if self.flatten_annotations:
            metadata = dict(example_id=example["qas"][0]["question_id"])
        else:
            metadata = dict(image_id=example["image_id"])
        return dict(
            image=join(DATA_HOME, example["image"]),
            message_list=message_list,
            metadata=metadata,
        )


class Vqa2(Ai2HfDataset):
    HF_SOURCE = "allenai/molmo2-vqa2-2014"

    @classmethod
    def download(cls, n_procs=1):
        _ensure_coco2014_downloaded()
        for k in ["train", "validation", "test"]:
            datasets.load_dataset_builder(cls.HF_SOURCE, name=k).download_and_prepare()

    def __init__(self, split: str, flatten_annotations=False, sample=None):
        super().__init__(split, self.HF_SOURCE, "vqa2", sample=sample,
                         flatten_annotations=flatten_annotations)


class TallyQa(Ai2HfDataset):
    HF_SOURCE = "allenai/molmo2-tally-qa"

    @classmethod
    def download(cls, n_procs=1):
        _ensure_coco2014_downloaded()
        _ensure_vg_downloaded()
        for k in ["train", "test"]:
            datasets.load_dataset_builder(cls.HF_SOURCE, name=k).download_and_prepare()

    def __init__(self, split: str, flatten_annotations=False, sample=None):
        super().__init__(split, self.HF_SOURCE, "tally_qa", sample=sample,
                         flatten_annotations=flatten_annotations)


class AOkVqa(Dataset):
    HF_SOURCE = "allenai/molmo2-a-ok-vqa"
    SPLITS = ["train", "val", "test"]

    @classmethod
    def download(cls, n_procs=1):
        _ensure_coco2017_downloaded()
        for k in ["train", "val", "test"]:
            datasets.load_dataset_builder(cls.HF_SOURCE, name=k).download_and_prepare()

    def __init__(self, split: str, direct_answer, flatten=False):
        self.data = datasets.load_dataset(self.HF_SOURCE, name=split, keep_in_memory=flatten)["train"]
        self.direct_answer = direct_answer
        if flatten:
            self.data = flatten_list([dict(message_list=[msg]) for msg in ex["message_list"]]
                                     for ex in self.data)
        self.split = split
        if self.split in ["validation", "test"] and direct_answer:
            assert flatten
            self.data = [x for x in self.data if not x["difficult_direct_answer"]]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = dict(self.data[item])
        metadata = {}
        for k in list(example):
            if k.startswith("metadata/"):
                metadata[k[len("metadata/"):]] = example.pop(k)
        if len(example["message_list"]) == 1:
            metadata["example_id"] = example["message_list"][0]["question_id"]
        example["metadata"] = metadata
        example["image"] = join(DATA_HOME, example["image"])

        messages = []
        for msg in example["message_list"]:
            if self.direct_answer:
                msg = dict(
                    question=msg["question"],
                    answers=msg["direct_answers"],
                    style="a_okvqa_da"
                )
            else:
                msg = dict(
                    question=msg["question"],
                    options=msg["choices"],
                    answer_idx=msg.get("answer_idx"),
                    style="a_okvqa_mc"
                )
            messages.append(msg)
        example["message_list"] = messages
        return example


class InfoQa(DatasetBase):
    home = join(DATA_HOME, "info_qa")
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                filename = "infographicsVQA_val_v1.0_withQT.json"
            else:
                filename = f"infographicsVQA_{split}_v1.0.json"
            if not exists(join(cls.home, filename)):
                raise ValueError(
                    "InfoQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 3)"
                    f" please download and unzip the data into `{cls.home}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            filename = "infographicsVQA_val_v1.0_withQT.json"
        else:
            filename = f"infographicsVQA_{split}_v1.0.json"
        filename = join(self.home, filename)
        log.info(f"Loading infoqa data from {filename}")
        with open(cached_path(filename, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            image_path = join(self.home, "images", ex.pop("image_local_name"))
            out.append(dict(
                image=image_path,
                question=ex["question"],
                answers=ex.get("answers", [""]),
                metadata=dict(example_id=ex["questionId"]),
            ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style="info_qa")


class DocQa(DatasetBase):
    home = join(DATA_HOME, "docqa")
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                split = "val"
            if split == "test":
                src = join(cls.home, f"{split}_v1.0.json")
            else:
                src = join(cls.home, f"{split}_v1.0_withQT.json")
            if not exists(src):
                import pdb; pdb.set_trace()
                raise ValueError(
                    "DocQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 1)"
                    f" please download and unzip the data into `{cls.home}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        if self.split == "test":
            src = join(self.home, f"{split}_v1.0.json")
        else:
            src = join(self.home, f"{split}_v1.0_withQT.json")
        log.info(f"Loading docqa data from {src}")
        with open(cached_path(src, cache_dir=environ.get("MOLMO_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            assert ex.pop("data_split") == split
            image_path = join(self.home, ex["image"])
            if self.split == "test":
                for k in ["answers", "question_types"]:
                    assert k not in ex
                    ex[k] = [""]
            out.append(dict(
                image=join(self.home, ex["image"]),
                question=ex["question"],
                answers=ex.get("answers"),
                metadata=dict(
                    doc_id=ex["docId"],
                    question_types=ex.get("question_types"),
                    example_id=ex["questionId"],
                ),
            ))
        return out

    def get(self, item, rng):
        return dict(self.data[item], style="doc_qa")


class SceneTextQa(DatasetBase):
    HOME = join(DATA_HOME, "scene-text")

    @classmethod
    def download(cls, n_procs=1):
        for split in ["train", "test"]:
            if not exists(join(join(cls.HOME, f"{split}_task_3.json"))):
                raise ValueError(
                    "SceneTextQa requires manually downloading https://rrc.cvc.uab.es/?ch=11"
                    f" please download and unzip the data into `{cls.HOME}`"
                )

    def __init__(self, split):
        assert split in ["train", "test", "validation"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "train"
        src = join(self.HOME, f"{self.split}_task_3.json")
        logging.info(f"Loading scene text data from {src}")
        data = json.loads(read_file(src))["data"]
        out = []
        for question in data:
            out.append(dict(
                image=join(self.HOME, question["file_path"]),
                question=question["question"],
                metadata=dict(example_id=question["question_id"]),
                answers=question.get("answers", []),
            ))
        if self.split in ["train", "validation"]:
            # Custom val split since the data doesn't have one
            out.sort(key=lambda x: x["metadata"]["example_id"])
            np.random.RandomState(63069).shuffle(out)
            if self.split == "train":
                return out[1024:]
            else:
                return out[:1024]
        else:
            return out

    def get(self, item, rng):
        return dict(self.data[item], style="st_qa")


class AI2D(Dataset):
    home = join(ACADEMIC_DATASETS, "academic_datasets", "ai2d")

    @classmethod
    def download(cls, n_procs=1):
        if exists(cls.home):
            return
        Ai2dDatasetBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "validation", "test"]:
            ds = Ai2dDatasetBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, cls.home, n_procs)

    def __init__(self, split, boxes="both", keep_in_memory=False):
        assert split in ["train", "validation", "test"]
        dataset = datasets.load_from_disk(
            self.home, keep_in_memory=keep_in_memory)[split]
        if boxes == "transparent":
            dataset = dataset.filter(lambda x: not x["abc_label"] or x["has_transparent_box"])
        elif boxes == "opaque":
            dataset = dataset.filter(lambda x: not x["abc_label"] or not x["has_transparent_box"])
        elif boxes == "both":
            pass
        else:
            raise NotImplementedError(boxes)
        self.dataset = dataset

        self.split = split
        self.boxes = boxes
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        ex = dict(
            image=_ex["image"],
            question=_ex["question"],
            answer_idx=_ex["correct_answer"],
            metadata=dict(
                example_id=_ex["question_id"],
                image_id=_ex["image_id"],
                abc_label=_ex["abc_label"],
                has_transparent_box=_ex["has_transparent_box"]
            ),
        )
        options = _ex["answer_texts"]
        if _ex["abc_label"] and sum(_ex["option_is_abc"]) >= (len(options)-1):
            ex["unlabelled_options"] = [
                opt.upper() if abc else opt
                for opt, abc in zip(options, _ex["option_is_abc"])
            ]
            ex["style"] = "ai2_diagram_no_letter"
        else:
            ex["options"] = options
            ex["style"] = "ai2_diagram"
        return ex


class PlotQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        PlotQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation", "test"]
        self.hf_dataset = PlotQaBuilder().as_dataset(split, in_memory=in_memory)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="plot_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class FigureQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "figure_qa")
        if exists(local_name):
            return
        FigureQaBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "validation1", "test1", "validation2", "test2"]:
            ds = FigureQaBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation1", "test1", "validation2", "test2"]
        self.hf_dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "figure_qa"), keep_in_memory=in_memory)[split]

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=str(a), style="figure_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class DvQa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "dv_qa")
        if exists(local_name):
            return
        DvQaBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "val_hard", "val_easy"]:
            ds = DvQaBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, in_memory=False):
        self.hf_dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "dv_qa"), keep_in_memory=in_memory)[split]

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="dv_qa"))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(image_id=example["image_id"]),
        )


class MathVista(HfDataset):
    PATH = "AI4Math/MathVista"

    @classmethod
    def download(cls, n_procs=None):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()

    def __init__(self, split, simplify_question=True, **kwargs):
        super().__init__(split, **kwargs)
        self.simplify_question = simplify_question

    def get(self, item, rng):
        ex = self.dataset[item]
        question: str = ex["question"]
        if self.simplify_question:
            question = question.split("Question:")[-1]
            question = question.split("Hint:")[0].strip()
        out = dict(
            question=question,
            image=ex["decoded_image"],
            metadata=dict(
                example_id=ex["pid"],
                answer=ex["answer"],
                precision=ex["precision"],
                query=ex["question"],
                choices=ex["choices"],
                question_type=ex["question_type"],
                answer_type=ex["answer_type"]
            ),
        )
        if ex["question_type"] == "multi_choice":
            out["options"] = ex["choices"]
            out["style"] = "eval_multiple_choice"
        else:
            out["style"] = "eval_short_answer"
        return out


class MMMU(Dataset):
    NAMES = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
        'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
        'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math',
        'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health',
        'Sociology'
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            if exists(join(DATA_HOME, "mmmu", name)):
                continue
            builder = datasets.load_dataset_builder("MMMU/MMMU", name=name)
            builder.download_and_prepare()

    def __init__(self, split: str, use_multi_image: bool = False):
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("MMMU/MMMU", name=name, split=split))
        self.data = datasets.concatenate_datasets(all_parts)
        self.use_multi_image = use_multi_image

    def __len__(self):
        return len(self.data)

    def replace_placeholders(self, all_strings: List[str]) -> List[str]:
        replaced = []
        for s in all_strings:
            replaced.append(re.sub(r"<image\s*(\d+)>", r"Image \1", s))

        return replaced

    def get(self, item, rng):
        ex = self.data[item]
        mc = ex["question_type"] == "multiple-choice"
        images = [ex[f"image_{i}"] for i in range(1, 8) if ex[f"image_{i}"] is not None]
        if len(images) > 1 and self.use_multi_image:
            style = "mantis_instruct_mc" if mc else "mantis_instruct_da"
            image = images
        else:
            style = 'a_okvqa_mc' if mc else 'vqa2'
            image = ex["image_1"]
        if self.use_multi_image:
            question = self.replace_placeholders([ex["question"]])[0]
        else:
            question = ex["question"]
        out = dict(
            image=image,
            text=ex["answer"],
            question=question,
            metadata=dict(answer=ex["answer"], example_id=ex["id"], question_type=ex["question_type"]),
            style=style
        )
        if mc:
            options = eval(ex["options"])
            if not self.use_multi_image and sum((re.match("<img='(.*?)'>", opt) is not None) for opt in options) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                del out["image"]
            elif self.use_multi_image:
                options = self.replace_placeholders(options)
            out["options"] = options
        return out


class RealWorldQa(HfDataset):
    PATH = "xai-org/RealworldQA"

    @classmethod
    def download(cls, n_procs=None):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()

    def __init__(self, mode="no_mc_instruction", in_memory=False):
        super().__init__("test", in_memory)
        self.mode = mode

    def get(self, item, rng):
        ex = self.dataset[item]
        prompt: str = ex["question"]
        if "Please answer directly with a single word or number." in prompt:
            question_type = "short_answer"
        else:
            assert "Please answer directly with only the letter of the correct option and nothing else." in prompt
            question_type = "multiple_choice"
        out = dict(
            image=ex["image"],
            metadata=dict(answer=ex["answer"], prompt=ex["question"], question_type=question_type),
        )
        if self.mode == "plain":
            out.update(style="none", prompt=prompt)
        else:
            if question_type == "short_answer":
                style = "eval_short_answer"
            else:
                style = "eval_multiple_choice"
            if self.mode == "no_instruction":
                if question_type == "short_answer":
                    prompt = prompt.split("\n")[0]
            else:
                if self.mode != "vqa_style_tag":
                    raise NotImplementedError(self.mode)
            out.update(style=style, question=prompt)
        return out


class CoSyn(Dataset):
    PATH = "allenai/CoSyn-400K"

    @classmethod
    def download(cls, n_procs=None):
        for config_name in ['chart', 'chemical', 'circuit', 'diagram', 'document', 'graphic',
                            'math', 'music', 'nutrition', 'table']:
            datasets.load_dataset_builder(cls.PATH, config_name).download_and_prepare()

    def __init__(self, doc_type: str, split: str, use_exp=True, keep_in_memory=False):
        self.doc_type = doc_type
        self.use_exp = use_exp
        self.split = split
        self.dataset = datasets.load_dataset(
            self.PATH, doc_type, split=split, keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        style = f"cosyn_{self.doc_type}"
        example = self.dataset[int(item)]
        qeas = example["qa_pairs"]
        if self.use_exp:
            style += "_exp"
            message_list = [
                dict(question=q, explanation=e, answer=a, style=style) for q, e, a in
                zip(qeas["question"], qeas["explanation"], qeas["answer"])
            ]
        else:
            message_list = [
                dict(question=q, answer=a, style=style) for q, a in
                zip(qeas["question"], qeas["answer"])
            ]
        return dict(
            image=example["image"],
            message_list=message_list,
            metadata=dict(
                image_id=example["id"]
            )
        )


class PixmoMulitDocQa(Dataset):
    PATH = "allenai/Molmo2-SynMultiImageQA"

    @classmethod
    def download(cls, n_procs=None):
        for config_name in ['chart', 'chemical', 'circuit', 'diagram',
                            'doc', 'graphic', 'music', 'table']:
            log.info(f"Downloading Cosyn-{config_name} config")
            datasets.load_dataset_builder(cls.PATH, config_name).download_and_prepare(num_proc=n_procs)

    def __init__(self, doc_type: str, split: str, use_exp=True,
                 keep_in_memory=False, max_images=5):
        self.doc_type = doc_type
        self.use_exp = use_exp
        self.max_images = max_images
        self.split = split
        data = datasets.load_dataset(
            self.PATH, doc_type, split=split, keep_in_memory=keep_in_memory)
        if max_images is not None:
            filtered = data.filter(
                lambda x: x["num_images"] <= self.max_images,
                input_columns=["metadata"]
            )
            logging.info(f"Filtered down to {len(filtered)} {doc_type} of {len(data)} docs with images<={max_images}")
            data = filtered
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        style = f"cosyn_{self.doc_type}"
        example = self.dataset[int(item)]
        qeas = example["qa_pairs"]
        if self.use_exp:
            style += "_exp"
            message_list = [
                dict(question=q, explanation=e, answer=a, style=style) for q, e, a in
                zip(qeas["question"], qeas["explanation"], qeas["answer"])
            ]
        else:
            message_list = [
                dict(question=q, answer=a, style=style) for q, a in
                zip(qeas["question"], qeas["answer"])
            ]
        return dict(
            image=example["images"],
            message_list=message_list,
            metadata=dict(
                image_id=example["id"]
            )
        )


class CoSynPoint(HfDataset):
    PATH = "allenai/CoSyn-point"

    def get(self, item, rng):
        example = self.dataset[item]
        messages = []
        for question, points, name in zip(
            example["questions"],
            example["answer_points"],
            example["names"]
        ):
            messages.append(dict(
                question=question,
                points=np.stack([points['x'], points['y']], -1),
                label=name,
                point_scale=100,
                style="cosyn_point",
            ))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_id=example["id"]
            )
        )


class ChartQa(HfDataset):
    """
    ChartQA dataset from HuggingFace M4 project. Can be weighted to balanced human/synthetic data
    """
    PATH = "HuggingFaceM4/ChartQA"

    def __init__(self, split: str, parts="both", weighted=False, keep_in_memory=False):
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]

        if split == "validation":
            split = "val"
        self.updated_split = split
        self.weighted = weighted
        self.parts = parts
        super().__init__(split, keep_in_memory=keep_in_memory)
        if self.parts != "both":
            # Filter out either human or aug datasets
            to_keep = 0 if (self.parts == "human") else 1
            self.dataset = self.dataset.filter(
                lambda x: x == to_keep,
                input_columns=["human_or_machine"]
            )

    def get(self, item, rng):
        ex = self.dataset[item]
        ex = dict(
            image=ex["image"],
            question=ex["query"],
            answers=ex["label"],
            style="chart_qa",
            metadata=dict(
                is_human=ex['human_or_machine'] == 0,
            )
        )
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex


class CountBenchQa(Dataset):

    @classmethod
    def download(self, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "countbench_qa")
        if exists(local_name):
            return
        CountQaBuilder().download_and_prepare()
        ds = CountQaBuilder().as_dataset("test")
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self):
        self.dataset = datasets.load_from_disk(join(ACADEMIC_DATASETS, "countbench_qa"))

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        return {
            'image': ex["image"],
            'question': ex['question'],
            'style': "point_count",
            'metadata': {
                'count': ex['count'],
                'image_id': ex["example_id"],
                'image_url': ex['image_url'],
            }
        }


class TabWMPDirectAnswer(Dataset):
    HOME = join(DATA_HOME, "academic_datasets", "tabwmp")

    @classmethod
    def download(cls, n_procs=1):
        if exists(cls.HOME):
            return
        TabMwpBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "dev", "test"]:
            ds = TabMwpBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, cls.HOME, n_procs)

    def __init__(self, split, include_options: bool, keep_in_memory=False):
        self.include_options = include_options
        self._dataset = datasets.load_from_disk(self.HOME, keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self._dataset)

    def get(self, item, rng):
        ex = self._dataset[item]
        out = dict(
            image=ex["image"],
            question=ex["question"],
            answer=ex["answer"],
            style="tabwmp_da",
            metadata=dict(
                example_id=ex["example_id"]
            )
        )
        if self.include_options and ex["choices"]:
            out["options"] = ex["choices"]
        return out


class PointBench(Dataset):
    SRC_DIR = join(DATA_HOME, "point_arena")
    IMAGES_DIR = Path(join(SRC_DIR, "selected_images"))
    MASKS_DIR = Path(join(SRC_DIR, "selected_masks"))

    @classmethod
    def download(cls, n_procs=1):
        for src_filename in ["data.json", "selected_images.zip", "selected_masks.zip"]:
            src_path = join(cls.SRC_DIR, src_filename)
            if not exists(src_path.replace(".zip", "")):
                log.info(f"Downloading {src_filename}...")
                os.makedirs(cls.SRC_DIR, exist_ok=True)
                downloaded_file = hf_hub_download(
                    repo_id="PointArena/pointarena-data", filename=src_filename, repo_type="dataset")
                if src_filename.endswith(".zip"):
                    # Using python to unzip mangles some filenames, so just use a subprocess call
                    subprocess.run(f'unzip {downloaded_file} -d {cls.SRC_DIR}', shell=True, check=True)

    @staticmethod
    def load_mask(mask_path):
        """Load a binary mask from a PNG file."""
        # Load the mask image
        mask_img = Image.open(mask_path)

        # Convert to numpy array (values will be 0 for black and 255 for white)
        mask_array = np.array(mask_img)

        # Normalize to binary (True/False) mask
        # For grayscale, consider any value > 127 as True
        if len(mask_array.shape) == 2:
            binary_mask = mask_array > 127
        # For RGB, consider any channel > 127 as True (if any channel is white)
        elif len(mask_array.shape) == 3:
            binary_mask = np.any(mask_array > 127, axis=2)
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")

        return binary_mask

    def __init__(self):
        self.data = self.load()

    def __len__(self):
        return len(self.data)

    def load(self):
        with open(join(self.SRC_DIR, "data.json"), "r") as f:
            data = json.load(f)
        return [x for x in data if "mask_filename" in x]

    def get(self, item, rng):
        item = self.data[item]
        image_filename = item["image_filename"]
        image_path = str(self.IMAGES_DIR / item["category"] / image_filename)
        mask_filename = item["mask_filename"]
        mask_path = str(self.MASKS_DIR / item["category"] / mask_filename)
        mask = self.load_mask(mask_path)
        query = item["user_input"]
        return dict(
            style="pointing",
            image=image_path,
            question=item["user_input"],
            metadata=dict(mask=mask, category=item["category"])
        )


class ScienceQAImageOnly(Dataset):
    """
    This class loads the ScienceQA dataset from HuggingFace (https://huggingface.co/datasets/derek-thomas/ScienceQA).
    """
    PATH = "derek-thomas/ScienceQA"

    @classmethod
    def download(self, n_procs=1):
        splits = None
        for split in ["train", "validation", "test"]:
            output_file = join(DATA_HOME, "science_qa_img_only", f"{split}.json")
            if exists(output_file):
                continue
            if splits is None:
                splits = datasets.load_dataset(self.PATH)
            ds = splits[split].cast_column("image", datasets.Image(decode=False))
            ds = ds.filter(lambda ex: ex["image"] is not None)
            image_ids = [f"{split}_{i+1}.png" for i in range(len(ds))]
            filenames = [
                join(DATA_HOME, "science_qa_img_only", "images", img_id)
                for img_id in image_ids
            ]
            image_iterator = (ex["image"]["bytes"] for ex in ds if ex["image"])
            saved_images = save_images(image_iterator, filenames, n_procs)
            examples = []
            for ex, img_id in zip(ds, image_ids):
                examples.append(dict(
                    image=img_id,
                    question=ex["question"],
                    hint=ex["hint"],
                    answer=ex["answer"],
                    choices=ex["choices"],
                ))
            write_json(output_file, examples)

    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        with open(resource_path(join(DATA_HOME, "science_qa_img_only", f"{split}.json"))) as f:
            self.data = json.load(f)
        super().__init__()

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        question =  ex["question"]
        hint = ex["hint"]
        if hint:
            question = hint + "\n" + question
        return dict(
            image=join(DATA_HOME, "science_qa_img_only", "images", ex["image"]),
            question=question,
            style="science_qa",
            answer_idx=ex["answer"],
            options=ex["choices"],
        )


def _get_gui_instruct_prompt(prompt_format, instruction):
    if prompt_format == "find":
        return "Find " + instruction
    elif prompt_format == "demo_find":
        return "demo: Find " + instruction
    elif prompt_format == "none":
        return instruction
    elif prompt_format == "click":
        return "Click " + instruction
    else:
        raise ValueError(f"Unknown prompt: {prompt_format}")


class ScreenSpotV2(Dataset):
    data_path = join(DATA_HOME, "ScreenSpot-v2")

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.data_path):
            log.info(f"Downloading ScreenSpot-v2...")
            snapshot_download(
                repo_id="OS-Copilot/ScreenSpot-v2",
                repo_type="dataset",
                local_dir=cls.data_path,
                local_dir_use_symlinks=False,
                max_workers=n_procs
            )
            image_zip_file = join(cls.data_path, "screenspotv2_image.zip")
            log.info(f"Extracting {image_zip_file}")
            with zipfile.ZipFile(image_zip_file, 'r') as zip_ref:
                zip_ref.extractall(cls.data_path)
            os.remove(image_zip_file)

    def __init__(self, kinds=("desktop", "web", "mobile"), prompt="find"):
        all_data = []
        self.prompt = prompt
        for kind in kinds:
            with open(join(self.data_path, f"screenspot_{kind}_v2.json")) as f:
                data = json.load(f)
                for ex in data:
                    ex["kind"] = kind
                all_data += [ex for ex in data]
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        metadata = {k: ex[k] for k in ["kind", "bbox", "data_type", "data_source", "instruction", "img_filename"]}
        return dict(
            instruction=ex["instruction"],
            img_filename=ex["img_filename"],
            image=join(self.data_path, "screenspotv2_image", ex["img_filename"]),
            prompt=_get_gui_instruct_prompt(self.prompt, ex["instruction"]),
            metadata=metadata,
            style="user_qa"
        )


class ScreenSpotPro(Dataset):
    data_path = join(DATA_HOME, "ScreenSpotPro")

    @classmethod
    def download(cls, n_procs=1):
        if not exists(cls.data_path):
            log.info(f"Downloading ScreenSpotPro...")
            snapshot_download(
                repo_id="likaixin/ScreenSpot-Pro",
                repo_type="dataset",
                local_dir=cls.data_path,
                local_dir_use_symlinks=False,
                max_workers=n_procs
            )

    def __init__(self, prompt="find"):
        all_data = []
        self.prompt = prompt
        for file in list_directory(join(self.data_path, "annotations")):
            with open(join(self.data_path, "annotations", file)) as f:
                all_data += json.load(f)
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        metadata = {k: ex[k] for k in ["bbox", "instruction", "id", "application", "platform", "ui_type", "group"]}
        return dict(
            image=join(self.data_path, "images", ex["img_filename"]),
            prompt=_get_gui_instruct_prompt(self.prompt, ex["instruction"]),
            metadata=metadata,
            style="user_qa"
        )


class OSWorldG(Dataset):
    HF_PATH = "MMInstruction/OSWorld-G"

    @classmethod
    def download(cls, n_procs=1):
        datasets.load_dataset_builder("MMInstruction/OSWorld-G").download_and_prepare(num_proc=n_procs)

    def __init__(self, prompt="none"):
        self.data = datasets.load_dataset(self.HF_PATH, split="test")
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        metadata = {k: ex[k] for k in ["box_coordinates", "instruction", "id", "box_type"]}
        return dict(
            image=ex["image"],
            prompt=_get_gui_instruct_prompt(self.prompt, ex["instruction"]),
            metadata=metadata,
            style="user_qa"
        )
