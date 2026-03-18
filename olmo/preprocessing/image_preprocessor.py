import dataclasses
import logging
import multiprocessing
import os
import warnings
from io import BytesIO
from typing import Tuple, Iterable, List, Set

import PIL
from PIL import ImageFile, ImageOps, Image
from tqdm import tqdm

from olmo.data.dataset import DATA_HOME, MULTI_IMG_DATA_HOME
from olmo.io import get_bytes_range, write_file, file_exists

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)


DEFAULT_IMAGE_PATH = "/weka/oe-training-default/mm-olmo/torch_datasets"
DEFAULT_MULTI_IMAGE_PATH = "/weka/oe-training-default/mm-olmo/multi_image_datasets"


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_pil_image(image_path: str) -> PIL.Image.Image:
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    # This a bit of hack to handle cases where the image path was hard-coded
    # into the dataset to the weka path
    if DATA_HOME != DEFAULT_IMAGE_PATH and DEFAULT_IMAGE_PATH in image_path:
        image_path = image_path.replace(DEFAULT_IMAGE_PATH, DATA_HOME)
    elif MULTI_IMG_DATA_HOME != DEFAULT_MULTI_IMAGE_PATH and DEFAULT_MULTI_IMAGE_PATH in image_path:
        image_path = image_path.replace(DEFAULT_MULTI_IMAGE_PATH, MULTI_IMG_DATA_HOME)
    # Ignore image loading warning
    with warnings.catch_warnings(record=True) as w:
        if image_path.startswith("gs://"):
            image_bytes = get_bytes_range(image_path, 0, None)
            return PIL.Image.open(BytesIO(image_bytes))
        else:
            return PIL.Image.open(image_path)


def load_image(image_path):
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    if isinstance(image_path, PIL.Image.Image):
        # Avoid annoying palette transparency warnings filling up the logs
        with warnings.catch_warnings(record=True) as w:
            image = image_path.convert("RGB")
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            pass
        return np.array(image)
    elif isinstance(image_path, np.ndarray):
        assert len(image_path.shape) == 3, "Image should have 3 dimensions"
        assert image_path.shape[2] == 3, "Image should have 3 channels"
        assert image_path.dtype == np.uint8, "Image should have uint8 type"
        return image_path
    else:
        # This a bit of hack to handle cases where the image path was hard-coded
        # into the dataset to the weka path
        if DATA_HOME != DEFAULT_IMAGE_PATH and DEFAULT_IMAGE_PATH in image_path:
            image_path = image_path.replace(DEFAULT_IMAGE_PATH, DATA_HOME)
        elif MULTI_IMG_DATA_HOME != DEFAULT_MULTI_IMAGE_PATH and DEFAULT_MULTI_IMAGE_PATH in image_path:
            image_path = image_path.replace(DEFAULT_MULTI_IMAGE_PATH, MULTI_IMG_DATA_HOME)

        # Ignore image loading warning
        with warnings.catch_warnings(record=True) as w:
            if image_path.startswith("gs://"):
                image_bytes = get_bytes_range(image_path, 0, None)
                with PIL.Image.open(BytesIO(image_bytes)) as image:
                    return load_image(image)
            else:
                with PIL.Image.open(image_path) as image:
                    return load_image(image)


def save_image(args) -> Tuple[str, bool]:
    image: PIL.Image.Image = args[0]
    filename: str = args[1]

    if isinstance(image, bytes):
        image_bytes = image
    else:
        assert isinstance(image, PIL.Image.Image), \
            f"{filename}: Expected a PIL image, got {type(image)}"
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    write_file(
        os.path.dirname(filename),
        os.path.basename(filename),
        image_bytes,
        save_overwrite=True
    )
    return filename, file_exists(filename)


def save_images(
    pil_images: Iterable[PIL.Image.Image],
    filenames: List[str],
    n_procs: int = 1,
) -> Set[str]:
    if n_procs != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_procs, initializer=setup_pil) as pool:
                for val in pool.imap_unordered(save_image, zip(pil_images, filenames)):
                    yield val
    else:
        setup_pil()
        def _iter():
            for val in zip(pil_images, filenames):
                yield save_image(val)

    pbar = tqdm(total=len(filenames), desc="Saving images")
    saved_images = set()
    for val in _iter():
        if val[1]:
            saved_images.add(val[0])
        pbar.update(1)
    pbar.close()
    logging.info(
        f"Saved {len(saved_images)}/{len(filenames)} ({len(saved_images)/len(filenames) * 100:0.2f}%) images")
    return saved_images


def resize_and_pad(
    image,
    desired_output_size,
    is_training=False,
    resize_method="torch-bilinear",
    pad_value=0,
    rng=np.random
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method in ["tensorflow", "tensorflow-random"]:
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        if resize_method == "tensorflow-random" and is_training:
            resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
            mode = resize_methods[rng.randint(len(resize_methods))]
            mode = getattr(tf.image.ResizeMethod, mode)
        else:
            mode = tf.image.ResizeMethod.BILINEAR
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=mode,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method in ["torch-bilinear", "torch-rng"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        if resize_method == "torch-rng"  and is_training:
            options = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST_EXACT,
                       InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING]
            mode = options[rng.randint(len(options))]
        else:
            mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def metaclip_resize(image, desired_output_size):
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    if torch.is_floating_point(image):
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
    else:
        assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 255)
        image = image / 255.0
    resized = torch.permute(image, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    return resized, image_mask


def siglip_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
    float32=True
) -> Tuple[np.ndarray, np.ndarray]:
    in_min = 0.0
    in_max = 255.0

    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    if float32:
        resized = resized.to(torch.float32)
        resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
        image_mask = None
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()
        image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def dino_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    else:
        assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0, 255).to(torch.float32)
        resized = resized / 255.0

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size,
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


@dataclasses.dataclass
class ImagePreprocessor:
    """Preprocesses an image, usually matching the pre-processing used by a pre-trained ViT"""
    normalize: str = "siglip"
    resize: str = "siglip"
    pad_value: float = 0
    image_patch_size: int = 14
    base_image_input_size: Tuple[int, int] = (336, 336)
    use_image_mask: bool = False
    normalize_on_gpu: bool = False

    def unnormalize_image(self, image: np.ndarray):
        if self.normalize_on_gpu:
            return image
        if self.normalize == "openai":
            return (image * np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :] +
                    np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :])
        elif self.normalize == "siglip":
            return (image + 1) / np.asarray(2.0, dtype=np.float32)
        elif self.normalize == "dino":
            return (image * np.array((0.229, 0.224, 0.225), dtype=np.float32)[None, None, :] +
                    np.array((0.485, 0.456, 0.406), dtype=np.float32)[None, None, :])
        else:
            raise NotImplementedError()

    def normalize_image_tensor(self, image: torch.FloatTensor):
        if self.normalize == "siglip":
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            return image * 2 - 1
        return image

    def normalize_image(self, image: np.ndarray):
        if self.normalize_on_gpu:
            return image
        if self.normalize == "openai":
            image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
            image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
        elif self.normalize == "siglip":
            image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
        elif self.normalize == "dino":
            image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
            image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        else:
            raise NotImplementedError(self.normalize)
        return image

    def resize_image(self, image, output_size, is_training, rng):
        if self.resize == "siglip":
            crop_arr, mask_arr = siglip_resize_and_pad(image, output_size, float32=not self.normalize_on_gpu)
        elif self.resize == "dino":
            crop_arr, mask_arr = dino_resize_and_pad(image, output_size)
        elif self.resize == "metaclip":
            crop_arr, mask_arr = metaclip_resize(image, output_size)
        else:
            resize = "torch-bilinear" if self.resize == "default" else self.resize
            crop_arr, mask_arr = resize_and_pad(
                image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training,
                resize_method=resize)
        return crop_arr, mask_arr

    def build_single_crop(self, image, is_training, rng, image_size=None):
        image_size = image_size or self.base_image_input_size
        resized, resized_mask = self.resize_image(image, image_size, is_training, rng)
        resized = self.normalize_image(resized)
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, 0)
        resized_mask = np.expand_dims(resized_mask, 0)
        crop_patch_w = image_size[1] // self.image_patch_size
        crop_patch_h = image_size[0] // self.image_patch_size
        resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
        if not self.use_image_mask:
            resized_mask = None
        return resized, resized_mask, resize_idx

    def build_vit_patch_mapping(self, original_image_w, original_image_h, max_crops, overlap_margins):
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size
        crop_size = base_image_input_size[0]
        assert base_image_input_size[0] == base_image_input_size[1]

        left_margin, right_margin = overlap_margins
        total_margin_pixels = image_patch_size*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * image_patch_size
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        crop_size = base_image_input_size[0]

        tiling = select_tiling(
            max(original_image_h - total_margin_pixels, 1),
            max(original_image_w - total_margin_pixels, 1),
            crop_window_size,
            max_crops
        )
        n_crops = tiling[0] * tiling[1]
        patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
        on = 0
        on_crop = 0
        for i in range(tiling[0]):
            # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
            # which results in overlapping crop windows
            y0 = i*crop_window_size
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
                patch_idx += on_crop * crop_patch_h * crop_patch_w

                # Mask out idx that are in the overlap region
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling[0]-1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling[1]-1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1
        return patch_idx_arr, tiling

    def build_overlapping_crops(self, image, is_training, rng, max_crops, overlap_margins):
        """Decompose an image into a set of overlapping crops

        :return crop_arr: [n_crops, h, w, 3] The crops
        :return mask_arr: [n_crops, h, w] The padding masks
        :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                           the crops were extracted from, what patch in `crop_arr` it corresponds to
        """
        original_image_h, original_image_w = image.shape[:2]
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size
        crop_size = base_image_input_size[0]
        assert base_image_input_size[0] == base_image_input_size[1]

        left_margin, right_margin = overlap_margins
        total_margin_pixels = image_patch_size*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * image_patch_size
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # Decide how to tile the image, to account for the overlap margins we compute the tiling
        # as if we had an image without the margins and were using a crop size without the margins
        tiling = select_tiling(
            max(original_image_h - total_margin_pixels, 1),
            max(original_image_w - total_margin_pixels, 1),
            crop_window_size,
            max_crops
        )
        src, img_mask = self.resize_image(
            image,
            [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
            is_training,
            rng
        )
        src = self.normalize_image(src)

        # Now we have to split the image into crops, and track what patches came from
        # where in `patch_idx_arr`
        n_crops = tiling[0] * tiling[1]
        crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
        mask_arr = np.zeros([n_crops, crop_size, crop_size], dtype=img_mask.dtype)
        patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
        on = 0
        on_crop = 0
        for i in range(tiling[0]):
            # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
            # which results in overlapping crop windows
            y0 = i*crop_window_size
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
                mask_arr[on_crop] = img_mask[y0:y0+crop_size, x0:x0+crop_size]
                patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
                patch_idx += on_crop * crop_patch_h * crop_patch_w

                # Mask out idx that are in the overlap region
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling[0]-1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling[1]-1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1

        # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
        # so it is ordered left-to-right order
        patch_idx_arr = np.reshape(
            patch_idx_arr,
            [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
        )
        patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
        patch_idx_arr = np.reshape(patch_idx_arr, [-1])

        # Now get the parts not in the overlap region, so it should map each patch in `src`
        # to the correct patch it should come from in `crop_arr`
        patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
            src.shape[0]//image_patch_size,
            src.shape[1]//image_patch_size,
            )
        if not self.use_image_mask:
            mask_arr = None
        return crop_arr, mask_arr, patch_idx_arr
