import dataclasses
from typing import List, Union, Tuple, Dict, Optional

import PIL
from PIL import ImageFile
from einops import einops
from olmo.config import BaseConfig

from olmo.preprocessing.image_preprocessor import ImagePreprocessor
from olmo.preprocessing.preprocessor_utils import TokenizedVisionData, batch_pixels_to_patches, \
    TensorSpec, TOKEN_POOLING_KEYS, arange_for_pooling
from olmo.tokenizer import HfTokenizerWrapper
import numpy as np

from transformers.image_utils import ImageInput


@dataclasses.dataclass
class MultiCropConfig(BaseConfig):
    """Config for preprocessing and tokenizing image(s)"""

    crop_mode: str = "resize"
    use_col_tokens: bool = True
    max_crops: int = 6
    high_res_max_crops: int = 24
    p_high_res: float = 0
    pooling_w: int = 2
    pooling_h: int = 2
    overlap_margins: Tuple[int, int] = (4, 4)
    max_images: Optional[int] = None
    max_multi_image_crops: int = 4
    multi_image_pooling_w: int = 2
    multi_image_pooling_h: int = 2
    use_single_crop_col_tokens: Optional[bool] = None
    use_single_crop_start_token: bool = False

    def build_image_preprocessor(self, tokenizer, image_preprocessor, image_padding_mask=False,
                                 legacy_image_mask=False, use_low_res_token_global_crops=False):
        image = MultiCropImagePreprocessor(
            tokenizer,
            image_preprocessor,
            legacy_image_mask,
            self.crop_mode,
            self.use_col_tokens,
            self.max_crops,
            self.high_res_max_crops,
            self.p_high_res,
            self.pooling_w,
            self.pooling_h,
            image_padding_mask,
            self.overlap_margins,
            use_low_res_token_global_crops=use_low_res_token_global_crops,
            use_single_crop_col_tokens=self.use_single_crop_col_tokens,
            use_single_crop_start_token=self.use_single_crop_start_token,
        )
        if self.max_images is not None:
            multi_image = MultiImagePreprocessor(
                dataclasses.replace(
                    image,
                    max_crops=self.max_multi_image_crops,
                    image_pooling_w=self.multi_image_pooling_w,
                    image_pooling_h=self.multi_image_pooling_h,
                    p_high_res=0,
                ),
                self.max_images
            )
        else:
            multi_image = None
        return image, multi_image


@dataclasses.dataclass
class MultiCropImagePreprocessor:
    """Converts an image into crops and tokens for the VLM"""

    tokenizer: HfTokenizerWrapper
    image_preprocessor: ImagePreprocessor
    legacy_image_mask: bool = False
    # How to crops/resize images
    crop_mode: str = "resize"
    use_col_tokens: bool = True
    max_crops: int = 6
    high_res_max_crops: int = 12
    p_high_res: 0 = 0
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    image_padding_mask: Union[bool, int] = False
    overlap_margins: Tuple[int, int] = (4, 4)

    use_low_res_token_global_crops: bool = False
    use_single_crop_col_tokens: Optional[bool] = None
    use_single_crop_start_token: bool = False


    def get_output_shapes(self) -> Dict[str, TensorSpec]:
        """Return the maximumly sized output shapes this preprocessor could produce"""
        specs = []
        # Computing the output shapes of this preprocessor statically is possible but
        # annoying to keep updated and can be error-prone
        # Instead we just preprocesses some large inputs and then retrieve the output shapes
        # col tokens and overlap regions means large rectangular images will produce the
        # largest possible output, so we use those images as input
        # HACK: `token_pooling` might have the larger first dimension for sqaure images,
        # so we need to add a square image to the specs
        for [h, w] in [[10000, 100], [100, 10000], [10000, 10000]]:
            tmp = self.p_high_res
            if tmp > 0:
                self.p_high_res = 1
            specs.append(TensorSpec.get_spec(self(np.zeros([h, w, 3], dtype=np.uint8))))
            if tmp > 0:
                self.p_high_res = tmp
        return TensorSpec.max_dictionaries(*specs)

    def __call__(
        self,
        image: ImageInput,
        is_training=False,
        rng=np.random,
    ) -> TokenizedVisionData:
        pooling_h = self.image_pooling_h
        pooling_w = self.image_pooling_w
        patch_id = self.tokenizer.image_patch_token_id
        base_image_input_size = self.image_preprocessor.base_image_input_size
        image_patch_size = self.image_preprocessor.image_patch_size
        if self.use_single_crop_col_tokens is None:
            use_single_crop_col_tokens = self.use_col_tokens
        else:
            use_single_crop_col_tokens = self.use_single_crop_col_tokens

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode == "resize":
            resized, resized_mask, resize_idx = self.image_preprocessor.build_single_crop(image, is_training=is_training, rng=rng)
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
            per_row = np.full(
                (w,),
                self.tokenizer.image_low_res_token_id if self.use_low_res_token_global_crops else patch_id,
                dtype=np.int32
            )
            if use_single_crop_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                [self.tokenizer.image_start_token_id],
                extra_tokens,
                [self.tokenizer.image_end_token_id],
            ]
            return TokenizedVisionData(
                tokens=np.concatenate(joint, 0),
                images=batch_pixels_to_patches(resized, image_patch_size),
                image_masks=None if resized_mask is None else batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1),
                token_pooling=pooling_idx
            )

        if self.crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
            if is_training and (self.p_high_res and rng.random() < self.p_high_res):
                max_crops = self.high_res_max_crops
            else:
                max_crops = self.max_crops
            crop_arr, mask_arr, patch_idx_arr = self.image_preprocessor.build_overlapping_crops(
                image, is_training=is_training, rng=rng,
                max_crops=max_crops, overlap_margins=self.overlap_margins)
            pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])

            # Now build the output tokens
            per_row = np.full(w, self.tokenizer.image_patch_token_id, dtype=np.int32)
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            joint = np.tile(per_row, [h])
            joint = [
                [self.tokenizer.image_start_token_id],
                joint,
                [self.tokenizer.image_end_token_id]
            ]

            if self.crop_mode == "overlap-and-resize":
                crop_arr = batch_pixels_to_patches(crop_arr, image_patch_size)
                if mask_arr is not None:
                    mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
                return np.concatenate(joint, 0), crop_arr, mask_arr, pooling_idx

            # Finally do the same for the global image
            resized, resized_mask, resize_idx = self.image_preprocessor.build_single_crop(image, is_training=is_training, rng=rng)
            crop_arr = np.concatenate([resized, crop_arr], 0)

            if mask_arr is not None:
                if self.legacy_image_mask:
                    mask_arr = np.pad(mask_arr.astype(np.float32), [[0, 1], [0, 0], [0, 0]], constant_values=-1)
                else:
                    mask_arr = np.concatenate([resized_mask, mask_arr], 0)

            resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = resize_idx.shape[:2]
            resize_idx = resize_idx.reshape([-1, pooling_h*pooling_w])

            # Global image goes first, so the order of patches in previous crops gets increased
            pooling_idx = np.where(
                pooling_idx >= 0,
                pooling_idx + crop_patch_h*crop_patch_w,
                -1
            )
            patch_idx_arr += crop_patch_h*crop_patch_w
            pooling_idx = np.concatenate([resize_idx, pooling_idx])

            per_row = np.full(
                (w,),
                self.tokenizer.image_low_res_token_id if self.use_low_res_token_global_crops else patch_id,
                dtype=np.int32
            )
            if use_single_crop_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            if self.use_single_crop_start_token:
                start_token = self.tokenizer.low_res_image_start_token_id
            else:
                start_token = self.tokenizer.image_start_token_id
            joint = [
                        [start_token],
                        extra_tokens,
                        [self.tokenizer.image_end_token_id],
                    ] + joint
            if mask_arr is not None:
                mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)

            cum_token_pooling_bounds = np.array([pooling_idx.shape[0]], dtype=np.int64)
            images = batch_pixels_to_patches(crop_arr, image_patch_size) 
            cum_image_bounds = np.array([images.shape[0]], dtype=np.int64)

            return TokenizedVisionData(
                tokens=np.concatenate(joint, 0),
                images=images,
                image_masks=mask_arr,
                token_pooling=pooling_idx,
                token_mapping=patch_idx_arr,
                cum_token_pooling_bounds=cum_token_pooling_bounds,
                cum_image_bounds=cum_image_bounds,
            )
        else:
            raise NotImplementedError(self.crop_mode)


@dataclasses.dataclass
class MultiImagePreprocessor:
    """Converts a list of images into crops and tokens for the VLM"""
    image_preprocessor: MultiCropImagePreprocessor
    max_images: int

    def get_output_shapes(self) -> Dict[str, TensorSpec]:
        image_shapes = self.image_preprocessor.get_output_shapes()
        tokens_shape = image_shapes.pop("tokens") * self.max_images
        tokens_shape = tokens_shape.extend(
            sum(len(self.image_preprocessor.tokenizer.encode(f"Image {i+1}")) for i in range(self.max_images))
        )
        for k, v in image_shapes.items():
            assert k in ["images", "image_masks", "token_pooling"]
            new_shape = list(v.shape)
            new_shape[0] *= self.max_images  # Only multiply batch dimension
            image_shapes[k] = TensorSpec(tuple(new_shape), v.dtype)
        image_shapes["tokens"] = tokens_shape
        return image_shapes

    def __call__(
        self,
        images,
        is_training=False,
        rng=None,
    ) -> List[TokenizedVisionData]:
        tokenized_images = []
        for idx, image in enumerate(images):
            image_data = self.image_preprocessor(image, is_training, rng)
            if len(images) > 1:
                # Add prefix to the image tokens if there are multiple images
                prefix = f"Image {idx + 1}"
                image_data.tokens = np.concatenate([self.image_preprocessor.tokenizer.encode(prefix), image_data.tokens], 0)
                assert image_data.position_ids is None
            tokenized_images.append(image_data)
            if self.max_images is not None and idx == self.max_images - 1:
                break
        return tokenized_images

