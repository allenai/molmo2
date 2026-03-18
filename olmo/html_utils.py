"""HTML utilities for visualizing datasets, preprocessing, or predictions"""
import base64
import io
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from os.path import join, exists, relpath
from typing import List, Dict, Any, Optional, Tuple

import PIL.Image
import numpy as np
from einops import einops
from html import escape as html_escape

from olmo import tokenizer
from olmo.preprocessing.image_preprocessor import load_image
from olmo.util import compute_hash
from olmo.preprocessing.point_formatter import extract_points, extract_multi_image_points, \
    extract_any_points
from olmo.tokenizer import get_special_token_ids

COLORS = [
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "gray",
    "green",
    "lime",
    "maroon",
    "navy",
    "olive",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow"
]


def unnormalize_image(image, normalize: str = "siglip"):
    """Normalizes the image to zero mean and unit variance."""
    if normalize == "siglip":
        image = (image + 1) / np.asarray(2.0, dtype=np.float32)
    elif normalize == "openai":
        image *= np.array((0.26862954, 0.26130258, 0.27577711))[None, None, :]
        image += np.array((0.48145466, 0.4578275, 0.40821073))[None, None, :]
    elif normalize == "dino":
        image *= np.array((0.229, 0.224, 0.225))[None, None, :]
        image += np.array((0.485, 0.456, 0.406))[None, None, :]
    else:
        raise NotImplementedError()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def escape_html(text):
    return "<br>".join(html_escape(x) for x in text.split("\n"))


def get_frame_coordinates_in_collage(x_orig, y_orig, orig_width, orig_height, target_size=128):
    """
    Transform coordinates from original image to resized image with padding.
    
    Args:
        x_orig: X coordinate in original image
        y_orig: Y coordinate in original image
        orig_width: Original image width 
        orig_height: Original image height 
        target_size: Target square size (default: 128)
    
    Returns:
        tuple: (x_new, y_new) coordinates in the resized image
    """
    # Calculate aspect ratio
    aspect_ratio = orig_width / orig_height
    
    # Determine new dimensions maintaining aspect ratio
    if orig_width > orig_height:
        # Width is limiting dimension
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
        pad_x = 0
        pad_y = (target_size - new_height) // 2
    else:
        # Height is limiting dimension
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        pad_x = (target_size - new_width) // 2
        pad_y = 0
    
    # Calculate scale factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    # Transform coordinates
    x_new = x_orig * scale_x + pad_x
    y_new = y_orig * scale_y + pad_y
    
    return (x_new, y_new)


def get_image_collage_coords_from_video_points(vid_points, video_w, video_h, fps=2, max_frames=128, num_frames_per_row=10, frame_size=128):
    """Convert video point text to coordinates

    Args:
        text: The text containing video points
        video_w: The width of the video
        video_h: The height of the video
        fps: The fps of the video and point annotations
        max_frames: The maximum number of frames
        num_frames_per_row: Number of frames per row in the collage
        frame_size: The size of each frame in the collage

    Returns: List of (x, y) coordinates in the collage
    """
    coords = []
    for triplet in vid_points:
        t, x, y = triplet
        frame_idx = int(t * fps)
        if frame_idx > max_frames - 1:
            continue
        row_idx = frame_idx // num_frames_per_row
        col_idx = frame_idx % num_frames_per_row
        frame_x, frame_y = get_frame_coordinates_in_collage(
            x, y, video_w, video_h, target_size=frame_size
        )
        new_x = col_idx * frame_size + frame_x
        new_y = row_idx * frame_size + frame_y
        coords.append((new_x, new_y))
    return coords

def get_fps_from_text(text):
    """Extract fps from video point text based on timestamp intervals
    
    Args:
        text: The text containing video points with timestamps (0.0 to 1.0)
        
    Returns:
        int: The fps value inferred from number of frames in 1 second
    """
    # Find all timestamps
    timestamps = re.findall(r'\b(\d+\.\d+)\s+<im_start>', text)
    
    if not timestamps:
        return 2  # Default fps
    
    # Convert to floats and filter to only 0.0 to 1.0 range
    timestamps_in_range = [float(t) for t in timestamps if float(t) < 1.0]
    
    # Number of unique frames in 1 second = fps
    fps = len(set(timestamps_in_range))
    
    return max(1, fps)  # Ensure at least 1 fps


def build_video_asset(video_src, src_folder, clip=None):
    assert isinstance(video_src, str)
    if clip is None:
        key = compute_hash(video_src)
        video_file = join(src_folder, key)
        if not exists(video_file):
            logging.info(f"Adding {video_file} to asset cache")
            shutil.copy(video_src, video_file)
    else:
        key = compute_hash(video_src + "|||" + str(clip))
        video_file = join(src_folder, key)
        if not exists(video_file):
            logging.info(f"Adding {video_file} {clip} to asset cache")
            import ffmpeg
            input_file = ffmpeg.input(video_src).trim(start=clip[0], end=clip[1]).setpts('PTS-STARTPTS')
            output_file = ffmpeg.output(input_file, video_file, format='mp4')
            ffmpeg.run(output_file)
    asset_dir = join(src_folder, "assets")
    os.makedirs(asset_dir, exist_ok=True)
    video_file = join(asset_dir, key)
    if not exists(video_file):
        logging.info(f"Adding {video_file} to asset cache")
        shutil.copy(video_src, video_file)
    return f'<video style="max-height: 448px; max-width: 448px" controls><source src="/assets/{key}"></video>'


def build_image_asset(image_src, src_folder=None):
    if src_folder is None:
        if isinstance(image_src, bytes):
            return f'data:image/jpeg;base64,{base64.b64encode(image_src).decode()}'
        elif isinstance(image_src, str):
            return build_embedded_image(load_image(image_src))
        else:
            return build_embedded_image(image_src)

    # Save the image in an "asset" dir and link to it
    if isinstance(image_src, str):
        with open(image_src, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_src, bytes):
        image_bytes = image_src
    elif isinstance(image_src, PIL.Image.Image) or isinstance(image_src, np.ndarray):
        if isinstance(image_src, np.ndarray):
            image_src = PIL.Image.fromarray(image_src)
        byte_stream = io.BytesIO()
        image_src.save(byte_stream, format='JPEG')
        image_bytes = byte_stream.getvalue()
    else:
        raise NotImplementedError(image_src)
    image_key = compute_hash(image_bytes)
    asset_dir = join(src_folder, "assets")
    os.makedirs(asset_dir, exist_ok=True)
    image_file = join(asset_dir, image_key)
    if not exists(image_file):
        logging.info(f"Adding {image_file} to asset cache")
        with open(image_file, 'wb') as f:
            f.write(image_bytes)
    return f"/assets/{image_key}"


def example_to_html_dict(ex, preprocessor, show_patches=False, show_crops=False, asset_cache=None):
    """Build HTML visualizations for an examples

    ex: The example (after preprocessing) to show
    preprocessor: The preprocessor used to preprocessor the examples
    show_patches: Whether to visualize the image features as patches
    show_crops: Whether to visualize crops used
    """
    if "metadata" in ex:
        metadata = ex["metadata"]
    else:
        metadata = {k[len("metadata/"):]: v for k, v in
                    ex.items() if k.startswith("metadata/")}
    voc = preprocessor.tokenizer

    # points, color, object ids
    all_points: List[Tuple[np.ndarray, str, Optional[np.ndarray]]] = []

    if "subsegment_ids" in ex:
        targets = ex["input_tokens"].ravel()
        subsegment_ids = ex["subsegment_ids"]
        shared_prefix = postprocess_prompt(voc.decode(targets[subsegment_ids == 1000]))
        segment_text = []
        for i in np.sort(np.unique(subsegment_ids)):
            if i == -1:
                continue
            mask = subsegment_ids == i
            segment_text.append((i, voc.decode(targets[mask], False), ex["loss_masks"][mask].mean()))

        text = []
        text.append("<ul>")
        text.append(str(ex['images'].shape))
        for i, raw_text, w in segment_text:
            seg = postprocess_prompt(raw_text)
            text.append("<li>")
            seg_points = np.array(extract_any_points(seg, metadata))
            if len(seg_points):
                color = COLORS[i % len(COLORS)]
                all_points.append((seg_points[:, 1:], color, seg_points[:, 0]))
                text.append(f"<span style=\"color: {color}\">SEGMENT {i}</span> w={w:0.3f}: " + escape_html(seg))
            else:
                text.append(f"SEGMENT {i} w={w:0.3f}: " + escape_html(seg))
            text.append("</li>")
        text.append("</ul>")
        text = " ".join(text)
    else:
        raw_text = voc.decode(ex["target_tokens"][ex["target_tokens"] != voc.pad_id], False)
        points = np.array(extract_any_points(raw_text, metadata))
        if len(points) > 0:
            all_points.append((points[:, 1:], "rgb(220,92,153)", points[:, 0]))
        text = escape_html(postprocess_prompt(raw_text))

    out = dict(text=text)
    out["num_tokens"] = (ex["target_tokens"] != -1).sum().item()

    if "masks" in metadata:
        # Show ground truth masks
        if "video" not in metadata and "points" not in metadata: # skip masks for video with point tracking for now
            img = np.any(metadata["masks"], 0).astype(np.uint8)
            max_dim = 768
            src = build_embedded_image(img*255)
            out["mask"] = f"<img style=\"max-height:{max_dim}px;max-width:{max_dim}px;height:auto;width:auto;\" src={src}><img>"

    if "image_paths" in metadata or "images" in metadata:
        # Show multi-image input
        if "image_paths" in metadata:
            images = metadata["image_paths"]
        else:
            images = metadata["images"]
        max_dim = 512
        images_html = []
        for image_ix, path in enumerate(images, start=0):
            src = build_image_asset(path, asset_cache)
            w, h = metadata["image_group_size"][image_ix]
            r = np.sqrt(h * w) / 40
            boxes = []
            for points, color, example_ids in all_points:
                img_points = points[points[:, 0] == image_ix]
                if len(img_points) > 0:
                    boxes.append(BoxesToVisualize(
                        [[x, y, r] for _, x, y in img_points], color, format="circle", shape="circle"))

            max_dim = 768
            if len(boxes) == 0:
                img_html = f"<img style=\"max-height:{max_dim}px;max-width:{max_dim}px;height:auto;width:auto;\" src={src}><img>"
            else:
                img_html = get_html_image_with_boxes(
                    src, boxes,
                    img_size=(w, h),
                    max_dim=max_dim
                )
            images_html.append(img_html)
        out["image"] = "".join(images_html)
    elif "video_frames" in metadata:
        # Show video input
        if "video_path" in metadata and asset_cache is not None:
            if "clip_start_time" in metadata:
                clip = (metadata["clip_start_time"], metadata["clip_end_time"])
            else:
                clip = None
            out["video"] = build_video_asset(metadata["video_path"], asset_cache, clip=clip)
        else:
            pass # TODO could show the frames in scrolling list of some sort

        if all_points:
            video_frames = metadata["video_frames"]
            frame_to_boxe = defaultdict(list)
            h, w = video_frames.frames.shape[1:3]
            r = np.sqrt(h * w) / 40
            for point_group, color, example_ids in all_points:
                if example_ids is not None and len(set(example_ids)) != len(example_ids):
                    for obj, (ts, x, y) in zip(example_ids, point_group):
                        frame_ix = np.argmin(np.abs(video_frames.timestamps - ts))
                        frame_to_boxe[frame_ix].append(BoxesToVisualize(
                            [[x, y, r]], COLORS[(int(obj)-1) % len(COLORS)], "circle", shape="circle"))
                else:
                    for ts, x, y in point_group:
                        frame_ix = np.argmin(np.abs(video_frames.timestamps - ts))
                        frame_to_boxe[frame_ix].append(BoxesToVisualize(
                            [[x, y, r]], color, "circle", shape="circle"))
            image_cells = []
            for frame_ix, boxes in sorted(frame_to_boxe.items(), key=lambda x: x[0]):
                frame = video_frames.frames[frame_ix]
                image_src = build_image_asset(frame, "/data/chrisc/viz")
                image_cells.append(get_html_image_with_boxes(image_src, boxes, max_dim=288, img_size=[w, h]))
            out["points"] = "\n".join(image_cells)
    elif "image" in metadata:
        image_src = build_image_asset(metadata["image"], asset_cache)
        w, h = metadata["image_size"]
        r = np.sqrt(h * w) / 40
        if image_src is not None:
            boxes = []
            for point_group, color, example_ids in all_points:
                if point_group.shape[-1] == 3:
                    assert np.all(point_group[:, 0] == 1)
                    point_group = point_group[:, 1:]
                boxes.append(BoxesToVisualize(
                    [[x, y, r] for x, y in point_group], color, "circle", shape="circle"))

            if image_src is not None:
                max_dim = 768
                if len(boxes) == 0:
                    out["image"] = f"<img style=\"max-height:{max_dim}px;max-width:{max_dim}px;height:auto;width:auto;\" src={image_src}><img>"
                else:
                    image_size = metadata.get("image_size")
                    out["image"] = get_html_image_with_boxes(
                        image_src, boxes,
                        img_size=image_size,
                        max_dim=max_dim
                    )

    if hasattr(preprocessor, "preprocessor"):
        image_preprocessor = preprocessor.preprocessor.image_preprocessor
    else:
        image_preprocessor = preprocessor.image_preprocessor
    vit_preprocessor = image_preprocessor.image_preprocessor
    patch_size = vit_preprocessor.image_patch_size
    base_h, base_w = vit_preprocessor.base_image_input_size
    if "images" in ex and (show_crops or show_patches):
        images = einops.rearrange(
            ex["images"], 't (h w) (dh dw c) -> t (h dh) (w dw) c',
            h=base_h//patch_size,
            w=base_w//patch_size,
            dh=patch_size, dw=patch_size, c=3)
        images = unnormalize_image(images)
    else:
        images = None

    if show_crops:
        n_crops = ex["images"].shape[0]
        crop_h, crop_w = base_h//patch_size, base_w//patch_size
        boxes_to_show = [[] for _ in range(len(images))]
        patches_used = []
        if ex.get("token_pooling") is not None:
            patches_used.append((
                {"border-color": "blue", "z-index": 100},
                ex["token_pooling"]
            ))
        if ex.get("low_res_token_pooling") is not None:
            patches_used.append((
                {"border-color": "red", "z-index": 110, "opacity": 0.7},
                ex["low_res_token_pooling"]
            ))

        for style, patch_id_set in patches_used:
            for patch_ids in patch_id_set:
                patch_ids = np.array(patch_ids)
                patch_ids = patch_ids[patch_ids >= 0]
                if len(patch_ids) == 0:
                    continue
                crop_ix = patch_ids.max() // (crop_h * crop_w)
                patch_ids %= (crop_h * crop_w)
                xs = (patch_ids % crop_h) * patch_size
                ys = (patch_ids // crop_h) * patch_size
                box = [xs.min(), ys.min(), xs.max()+patch_size, ys.max()+patch_size]
                boxes_to_show[crop_ix].append(BoxesToVisualize(np.array([box]), style=style))

        all_crops = []
        for crop_ix, (crop, boxes) in enumerate(zip(images, boxes_to_show)):
            all_crops.append(get_html_image_with_boxes(build_embedded_image(crop), boxes))
        out[f"crops"] = "\n".join(all_crops)

    if show_patches:
        pooled_patches_idx = ex["pooled_patches_idx"]
        special_token_to_id = get_special_token_ids(voc)
        image_patch_id = special_token_to_id[tokenizer.IMAGE_PATCH_TOKEN]
        id_to_special_token = {i: k for i, k in special_token_to_id.items()}
        with_patches = []
        patches = einops.rearrange(images,
                                   't (h dh) (w dw) c -> (t h w) dh dw c',
                                   dh=patch_size, dw=patch_size
                                   )
        on_pooled_patch = 0
        for token_ix, ix in enumerate(ex["input_tokens"]):
            if ix == -1:
                with_patches.append("<PAD>")
            elif ix == image_patch_id:
                # [pool_h, pool_w, patch_h, patch_w, dim]
                sub_patches = patches[pooled_patches_idx[on_pooled_patch]]
                patch = einops.rearrange(
                    sub_patches,
                    '(pool_h pool_w) patch_h patch_w c -> (pool_h patch_h) (pool_w patch_w) c',
                    pool_h=preprocessor.mm_preprocessor.high_res_pooling_h,
                    pool_w=preprocessor.mm_preprocessor.high_res_pooling_w
                )

                src = build_embedded_image(patch)
                with_patches.append(f"<img src={src}></img>")
                on_pooled_patch += 1
            elif ix in id_to_special_token:
                with_patches.append(html_escape(str(id_to_special_token)))
            else:
                with_patches.append(html_escape(voc.decode([ix])))
        out["tokens"] = " ".join(with_patches)

    return out


def build_embedded_image(image_data):
    """Turns an image into a string that can be used as a src in html images"""
    if image_data.dtype == np.float32:
        image_data = (image_data*255).astype(np.uint8)
    with PIL.Image.fromarray(image_data) as img:
        image_data = io.BytesIO()
        img.save(image_data, format='JPEG')
        image_data = image_data.getvalue()
    encoded_image = base64.b64encode(image_data)
    return f'data:image/jpeg;base64,{encoded_image.decode()}'


def build_html_table(data: List[Dict[str, Any]], col_widths=None, fixed_width=False) -> str:
    columns = {}  # Collect any key that appears in the data, in order
    for row in data:
        for key in row:
            columns[key] = None
    html = [
"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta content="text/html;charset=utf-8" http-equiv="Content-Type">
  <meta content="utf-8" http-equiv="encoding">
</head>
""".strip()
    ]

    # Body
    html.append("<body>")
    if fixed_width:
        html.append("<table style=\"table-layout: fixed; width:100%\">")
    else:
        html.append("<table>")

    # Table Header
    html.append("<tr>")
    header = []
    for c in columns:
        if col_widths and c in col_widths:
            header.append(f"<th style=\"{col_widths[c]}\">{c}</th>")
        else:
            header.append(f"<th>{c}</th>")
    html.append(" ".join(header))
    html.append("</tr>")

    # Table Body
    for ex in data:
        cells = []
        for c in columns:
            val = ex.get(c)
            if val is None:
                cells.append("")
            elif isinstance(val, str):
                cells.append(val)
            elif isinstance(val, (float, int)):
                cells.append(val)
            elif len(val.shape) == 3 and val.shape[-1] == 3:
                # Assume an image
                data = build_embedded_image(val)
                cells.append(f'<img src={data}></img>')
            else:
                raise NotImplementedError(f"Data not understood for {val.shape}: {val}")
        html.append("<tr>")
        html.append("\n".join(f"<td>{x}</td>" for x in cells))
        html.append("</tr>")

    html.append("</table>")
    html.append("</body>")
    html.append("</html>")
    return "\n".join(html)


@dataclass
class BoxesToVisualize:
    """Boxes to draw on an image"""
    boxes: Any
    color: str=None
    format: str = "yxyx"
    labels: List[str] = None
    shape: str = "box"
    style: Dict[str, Any] = None
    scores: Optional = None


def html_rect(x1, y1, x2, y2, style=None, score=None, label=None, text_color=None,
              shape="box"):
    """Utility method to get a HTML rectangle element"""
    rect_style = {
        "position": "absolute",
        "top": f"{y1}px",
        "left": f"{x1}px",
        "height": f"{y2-y1}px",
        "width": f"{x2-x1}px",
    }
    rect_style.update(style)
    rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

    text_style = {
        "position": "absolute",
        "top": y1-5,
        "left": x1+3,
        "color": text_color,
        "background-color": "black",
        "z-index": 9999,
        "padding-right": "5px",
        "padding-left": "5px",
    }
    text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

    if label is None:
        text = ''
    else:
        text = f'  <div style="{text_style_str}">{label}</div>'

    if text:
        html = [f'<span style="{rect_style_str}"></span>']
    else:
        html = [
            f'<div>',
            f'  <div style="{rect_style_str}"></div>',
            text,
            "</div>"
        ]
    return html


def get_html_image_with_boxes(
    image_src, boxes: List[BoxesToVisualize], width=None, height=None, wrap="div",
    img_size=None, max_dim=None, image_style=None) -> str:
    """Build a HTML element containing `image_src` and the boxes in `boxes` on top of it.

    Provides a way to draw annotated images without have to load/modify the image itself
    """
    html = []
    html += [f'<{wrap} style="display: inline-block; position: relative;">']
    image_attr = dict(src=image_src)
    if image_style:
        image_attr["style"] = image_style
    if max_dim is not None:
        assert height is None and width is None
        scale = max_dim / max(img_size)
        if scale > 0:
            width = round(img_size[0]*scale)
            height = round(img_size[1]*scale)

    if width:
        image_attr["width"] = width
    if height:
        image_attr["height"] = height
    attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
    html += [f'<img {attr_str}>']

    for box_set in boxes:
        if height or width:
            img_w, img_h = img_size
            if not width:
                factor = height/img_h
                w_factor = factor
                h_factor = factor
            elif height and width:
                w_factor = width/img_w
                h_factor = height/img_h
            else:
                raise NotImplementedError()
        else:
            w_factor = 1
            h_factor = 1

        if boxes is not None and len(boxes) > 0:
            task_boxes = np.asarray(box_set.boxes)
            if box_set.format == "yxyx":
                task_boxes = np.stack([
                    task_boxes[:, 1], task_boxes[:, 0],
                    task_boxes[:, 3], task_boxes[:, 2],
                ], -1)
            elif box_set.format == "xyxy":
                pass
            elif box_set.format == "circle":
                r = task_boxes[:, 2]
                task_boxes = np.stack([
                    task_boxes[:, 0] - r,
                    task_boxes[:, 1] - r,
                    task_boxes[:, 0] + r,
                    task_boxes[:, 1] + r
                ], -1)
            elif box_set.format == "xywh":
                task_boxes = np.stack([
                    task_boxes[:, 0], task_boxes[:, 1],
                    task_boxes[:, 0] + task_boxes[:, 2],
                    task_boxes[:, 1] + task_boxes[:, 3]
                ], -1)
            else:
                raise NotImplementedError(box_set.format)

        for ix in range(len(task_boxes)):
            box = task_boxes[ix]
            x1, y1, x2, y2 = box
            rect_style = {}

            if box_set.shape == "box":
                rect_style = {
                    "border-style": "solid",
                    "border-color": box_set.color,
                    "box-sizing": "border-box",
                }
            elif box_set.shape == "box_full":
                rect_style = {
                    "background-color": box_set.color,
                }
            elif box_set.shape == "circle":
                rect_style = {
                    "background-color": box_set.color,
                    "border-radius": "50%",
                }
            else:
                raise NotImplementedError(f"Shape not understood: {box_set.shape}")
            if box_set.style is not None:
                rect_style.update(box_set.style)

            html += html_rect(
                x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor,
                style=rect_style,
                label=None if box_set.labels is None else box_set.labels[ix],
                score=None if box_set.scores is None else box_set.scores[ix],
            )

    html += [f'</{wrap}>']
    return "\n".join(html)


def postprocess_prompt(prompt_text, show_col_tokens=False):
    """Get a human-readable prompt by compressing the image tokens"""
    start = 0
    prompt_text = prompt_text.lstrip()  # some tokenizers add a leading space before special tokens
    post_processed_text = ""
    target_tokens = [tokenizer.IMAGE_LOW_RES_TOKEN, tokenizer.IMAGE_PATCH_TOKEN]
    if not show_col_tokens:
        target_tokens.append(tokenizer.IM_COL_TOKEN)
    target_re = "|".join(target_tokens) + r"\s?"
    for match in re.finditer(fr"({target_re})+", prompt_text):
        n_patches = sum(match.group(0).count(x) for x in target_tokens)
        if match.start() > start:
            post_processed_text += prompt_text[start:match.start()]
        prefix = next(re.finditer(target_re, match.group(0))).group(0)
        post_processed_text += f"{prefix}[{n_patches}]"
        start = match.end()

    post_processed_text += prompt_text[start:]
    return post_processed_text

