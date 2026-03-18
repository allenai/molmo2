import dataclasses
import math
from typing import Optional, Any, List, Union, Dict

import numpy as np
from olmo.config import BaseConfig

from olmo import tokenizer
from olmo.models.molmo_point.molmo_point_data_formatter import Message
from olmo.preprocessing.text_preprocessor import MessageWeight, build_subsegment_pos_ids

ATTEND_ALL_SUBSEGMENT_ID = 10000


@dataclasses.dataclass
class MolmoPointTextPreprocessorConfig(BaseConfig):
    max_answer_len: Optional[int] = None
    last_message_loss_only: bool = False
    max_text_tokens: Optional[str] = None
    loss_token_weighting: Optional[str] = None

    def build_text_preprocessor(self, tokenizer, max_seq_len):
        if self.loss_token_weighting == "root_subsegments":
            weighting = MessageWeight(root_subsegments=True)
        elif self.loss_token_weighting == "root_subsegments_root_tokens":
            weighting = MessageWeight(root_subsegments=True, root_length=True)
        elif self.loss_token_weighting is None:
            weighting = MessageWeight()
        else:
            raise NotImplementedError(self.loss_token_weighting)
        return MolmoPointInterleavedTextPreprocessor(
            tokenizer,
            self.max_text_tokens,
            max_seq_len,
            self.last_message_loss_only,
            self.max_answer_len,
            default_message_weight=weighting
        )


@dataclasses.dataclass
class MolmoPointInterleavedTextPreprocessor:
    """
    Build batches from text that is interleaved with tokens from other modalities

    This preprocessor also collects and aggregates points that are attatched to each message
    """
    tokenizer: Any = None
    max_text_tokens: Optional[int] = None
    max_sequence_length: Optional[int] = None
    last_message_loss_only: bool = False
    max_answer_len: int = None
    default_message_weight: Optional[MessageWeight] = dataclasses.field(default_factory=MessageWeight)

    def tokenize_message(self, message_list: List[Message], weight, bos=True, add_last_eos=True):
        if bos:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            text_token_ids = [bos]
            loss_mask = [0.0]
        else:
            text_token_ids = []
            loss_mask = []

        point_target_ids = []
        for msg_ix, message in enumerate(message_list):
            message_ids = self.tokenizer.encode(message.text)
            is_model = msg_ix % 2 == 1
            if message.points is not None and len(message.points) > 0:
                if not is_model:
                    raise NotImplementedError("User points")
                else:
                    if (message.points[:, 0] >= 0).sum() != (np.array(message_ids) == self.tokenizer.token_index_token_id).sum():
                        raise RuntimeError(f"Message had mis-matched point tokens/token labels: {message}")
                    point_target_ids.append(message.points)

            if is_model and (add_last_eos or msg_ix != len(message_list) - 1):
                message_ids.append(self.tokenizer.eos_token_id)

            if is_model and self.max_answer_len:
                message_ids = message_ids[:self.max_answer_len]

            has_loss = is_model and (
                not self.last_message_loss_only or (msg_ix == (len(message_list) - 1)))
            loss_mask += [has_loss] * len(message_ids)
            text_token_ids += message_ids
        text_token_ids = np.array(text_token_ids)
        loss_mask = np.array(loss_mask, dtype=np.float32)
        if weight.root_length:
            if loss_mask.sum() > 0:
                loss_mask *= 2 / np.sqrt(loss_mask.sum())
        if weight.weight is not None:
            loss_mask *= weight.weight
        point_target_ids = np.concatenate(point_target_ids) if point_target_ids else None
        return text_token_ids, loss_mask, point_target_ids

    def tokenize_message_list(
        self,
        message_list: List[List[Message]],
        n_mm_tokens: int,
        num_images: int = 1,
        weights: List[MessageWeight] = None,
    ):
        """Handle multi-annotation data where we have many annotations for one multi-modal input"""
        assert len(message_list) > 0, "Given empty messages"
        # Multi-annotation data where we have many annotations for one multi-modal input
        before_ids = []
        after_ids = []
        before_losses = []
        after_losses = []
        before_subsegments = []
        after_subsegments = []
        point_target_ids = []
        n_tokens = 0
        for message_set_ix, message_tuple in enumerate(message_list):
            add_bos = message_set_ix == 0
            weight = weights[message_set_ix]
            text_ids, text_loss, point_targets = self.tokenize_message(
                message_tuple, weight, bos=add_bos, add_last_eos=False)
            is_prompt = text_ids == self.tokenizer.image_prompt_token_id
            n_prompts = is_prompt.sum()
            if n_prompts == 1:
                image_idx = np.argmax(is_prompt)
                s, e = image_idx, image_idx+1
            elif n_prompts == 0 and add_bos:
                s, e = 1, 1
            elif n_prompts == 0:
                s, e = 0, 0
            else:
                raise NotImplementedError("Multi-message with multi images")

            if text_loss[e] != 0:
                raise ValueError("Must have a non-loss token after MM data")
            if self.max_sequence_length and message_set_ix != 0:
                if (n_mm_tokens + n_tokens + np.argmax(text_loss != 0)) >= self.max_sequence_length:
                    # This example would get no loss tokens anyway
                    break

            if point_targets is not None:
                if n_prompts != 0:
                    raise ValueError("Interleaved image/text and point labels")
                point_target_ids.append(point_targets)

            n_tokens += len(text_ids)
            before_ids.append(text_ids[:s])
            after_ids.append(text_ids[e:])
            before_losses.append(text_loss[:s])
            after_losses.append(text_loss[e:])
            before_subsegments.append(np.full(s, ATTEND_ALL_SUBSEGMENT_ID, dtype=np.int32))
            after_subsegments.append(np.full(len(text_ids[e:]), message_set_ix, dtype=np.int32))
            if self.max_text_tokens and (n_tokens >= self.max_text_tokens):
                break

        # Account for root_subsegments, we do this here so it does not account for
        # messages that were truncated
        for ix, weight in enumerate(weights[:len(before_ids)]):
            if weight.root_subsegments:
                before_losses[ix] /= math.sqrt(len(before_ids))
                after_losses[ix] /= math.sqrt(len(before_ids))

        text_token_ids = np.concatenate([
            np.concatenate(before_ids),
            [self.tokenizer.image_prompt_token_id] * num_images,
            np.concatenate(after_ids),
            [self.tokenizer.eos_token_id],
            ])
        text_subsegments = np.concatenate([
            np.concatenate(before_subsegments),
            [ATTEND_ALL_SUBSEGMENT_ID] * num_images,
            np.concatenate(after_subsegments),
            after_subsegments[-1][-1:]  # for EOS
        ])
        text_loss_masks = np.concatenate([
            np.concatenate(before_losses),
            [0] * num_images,
            np.concatenate(after_losses),
            [0]  # for EOS
        ])
        point_target_ids = np.concatenate(point_target_ids, 0) if point_target_ids else None
        return text_token_ids, text_loss_masks, text_subsegments, point_target_ids

    def _maybe_truncate_example(self, ex):
        if self.max_sequence_length and (self.max_sequence_length < len(ex["input_tokens"])):
            # When truncating, we have to be careful to also truncate the
            # number of point target ids
            if np.any(ex["input_tokens"][self.max_sequence_length:] == self.tokenizer.image_patch_token_id):
                raise ValueError()
            for k, v in ex.items():
                ex[k] = v[:self.max_sequence_length]
            if "point_target_ids" in ex:
                points_targets = ex.pop("point_target_ids")
                input_ids = ex["input_tokens"]
                n_patch = (input_ids == self.tokenizer.token_index_token_id).sum()
                n_subpatch = (input_ids == self.tokenizer.subpatch_index_token_id).sum()
                points_targets = points_targets[:n_patch]

                # Some subpatch/location IDs might already be negative due to ending
                # messages with "no more point" patch token ids, so remove the first
                # non-negative ones
                subpatches_removed = (points_targets[:, 1] >= 0).sum() - n_subpatch
                assert subpatches_removed >= 0
                if subpatches_removed > 0:
                    non_neg_indices = np.argwhere(points_targets[:, 1] != -1)
                    points_targets[non_neg_indices[-subpatches_removed:], 1] = -1
                if points_targets.shape[1] == 3:
                    n_loc = (input_ids == self.tokenizer.subpatch_loc_token_id).sum()
                    loc_removed = (points_targets[:, 2] >= 0).sum() - n_loc
                    assert loc_removed >= 0
                    if loc_removed > 0:
                        assert loc_removed - subpatches_removed in [0, 1]
                        non_neg_indices = np.argwhere(points_targets[:, 2] != -1)
                        points_targets[non_neg_indices[-loc_removed:], 2] = -1
                ex["point_target_ids"] = points_targets

        # Sanity check
        input_ids = ex["input_tokens"]
        if "point_target_ids" in ex:
            points_targets = ex["point_target_ids"]
            n_patch = (input_ids == self.tokenizer.token_index_token_id).sum()
            n_subpatch = (input_ids == self.tokenizer.subpatch_index_token_id).sum()
            assert np.all(points_targets[:, 0] >= 0)
            assert np.all(points_targets[:, 0] <= 1000000)
            assert len(points_targets) == n_patch
            assert (points_targets[:, 1] >= 0).sum() == n_subpatch
            if points_targets.shape[1] == 3:
                n_loc = (input_ids == self.tokenizer.subpatch_loc_token_id).sum()
                assert (points_targets[:, 2] >= 0).sum() == n_loc
        return ex

    def tokenize_and_interleave(
        self,
        message_list: List[List[Message]],
        multi_model_tokens: List[np.ndarray],
        multi_model_pos_ids: Optional[List[np.ndarray]]=None,
        weight: Optional[float]=None
    ) -> Dict[str, np.ndarray]:
        """
        Build a batch by interleaving the text tokens from tokenizing `message_list` and the
        multi-modal tokens from `multi_model_tokens`

        `tokenizer.IMAGE_PROMPT` is used to show where the MM tokens should be inserted, if it is
        not present the MM tokens are inserted right after BOS

        If `message_list` is a list of lists, the batch is assumed to contain multiply-annotated
        MM data. The batch will include tokens from all messages but the MM tokens only once, and
        `subsegment_id` will indicate how to cross-attend between the tokens. Attending between
        tokens before the MM tokens will be allowed, but attending between tokens after the MM
        tokens will not.
        """
        if not isinstance(weight, (list, tuple)):
            weight = [self.default_message_weight.with_overrides(weight)] * len(message_list)
        else:
            weight = [self.default_message_weight.with_overrides(x) for x in weight]
        if len(message_list) == 1:
            text_token_ids, text_loss_masks, point_target_ids = self.tokenize_message(message_list[0], weight[0])
            text_subsegments = None
            for_inference = len(message_list[0]) % 2 == 1
        else:
            text_token_ids, text_loss_masks, text_subsegments, point_target_ids = self.tokenize_message_list(
                message_list, sum(len(x) for x in multi_model_tokens),
                0 if multi_model_tokens is None else len(multi_model_tokens),
                weight
            )
            for_inference = False

        if len(multi_model_tokens) > 0:
            mm_idx = np.argwhere(text_token_ids == self.tokenizer.image_prompt_token_id)
            if len(mm_idx) == 0:
                if multi_model_tokens is not None:
                    # Assume mm data should go right after BOS
                    mm_idx = [1] * len(multi_model_tokens)
            else:
                mm_idx = mm_idx[:, 0]
        else:
            mm_idx = []

        mm_tokens = []
        mm_loss_masks = []
        mm_subsegments = None if text_subsegments is None else []
        mm_position_ids = []
        on = 0
        on_pos = 0
        for i, token_ix in enumerate(mm_idx):
            mm_tokens.append(text_token_ids[on:token_ix])
            mm_loss_masks.append(text_loss_masks[on:token_ix])
            if text_subsegments is not None:
                mm_subsegments.append(text_subsegments[on:token_ix])
            if multi_model_pos_ids is not None:
                assert len(multi_model_tokens[-1]) == len(multi_model_pos_ids[-1])
                mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
                on_pos += len(mm_tokens[-1])

            vision_tokens = multi_model_tokens[i]
            mm_tokens.append(vision_tokens)
            mm_loss_masks.append(np.zeros_like(vision_tokens))
            if text_subsegments is not None:
                mm_subsegments.append(np.full([len(vision_tokens)], text_subsegments[token_ix]))
            if multi_model_pos_ids is not None:
                mm_position_ids.append(multi_model_pos_ids[i] + on_pos)
                on_pos += multi_model_pos_ids[i].max() + 1
            if text_token_ids[token_ix] == self.tokenizer.image_prompt_token_id:
                on = token_ix + 1  # Skip over the image prompt token
            else:
                on = token_ix

        mm_tokens.append(text_token_ids[on:])
        mm_loss_masks.append(text_loss_masks[on:])
        if text_subsegments is not None:
            mm_subsegments.append(text_subsegments[on:])
            n_pre_mm_tokens = sum(len(x) for x in mm_tokens[:-1])
            if not mm_position_ids:
                mm_position_ids = [np.arange(n_pre_mm_tokens)]
                on_pos = n_pre_mm_tokens
            mm_position_ids.append(on_pos + build_subsegment_pos_ids(text_subsegments[on:]))
        elif mm_position_ids:
            mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
        else:
            mm_position_ids = [np.arange(0, sum(len(x) for x in mm_tokens))]

        mm_tokens = np.concatenate(mm_tokens)
        mm_loss_masks = np.concatenate(mm_loss_masks)
        mm_position_ids = np.concatenate(mm_position_ids)
        if mm_subsegments is not None:
            mm_subsegments = np.concatenate(mm_subsegments)

        target_tokens = mm_tokens

        if not for_inference:
            target_tokens = mm_tokens[1:]
            input_tokens = mm_tokens[:-1]
            mm_loss_masks = mm_loss_masks[1:]
            if mm_subsegments is not None:
                # The targets for subsegments in the middle need to end with EOS,
                # currently they end with whatever starts the next segment
                mm_subsegments = mm_subsegments[:-1]
                target_tokens = np.copy(target_tokens)
                for i in range(len(message_list)):
                    subsegment_mask = mm_subsegments == i
                    if not np.any(mm_subsegments == i):
                        assert (self.max_text_tokens or self.max_sequence_length) and i != 0
                        # Message skipped due hitting `self.max_text_tokens`
                        break
                    segment_end = np.argwhere(mm_subsegments == i)[-1, 0]
                    target_tokens[segment_end] = self.tokenizer.eos_token_id
                    assert mm_subsegments[segment_end-1] == i
                    assert mm_loss_masks[segment_end-1] != 0
                    mm_loss_masks[segment_end] = mm_loss_masks[segment_end-1]

            if mm_loss_masks[-1] == 0:
                raise RuntimeError("EOS should not be masked")
            mm_position_ids = mm_position_ids[:-1]
        else:
            # Presumably doing inference, but give a dummy target anyway for consistency
            assert mm_tokens[-1] != self.tokenizer.eos_token_id
            input_tokens = mm_tokens
            target_tokens = np.pad(mm_tokens[1:], [0, 1], constant_values=0)

        out = {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "loss_masks": mm_loss_masks,
            "position_ids": mm_position_ids,
        }
        if mm_subsegments is not None:
            out["subsegment_ids"] = mm_subsegments

        # Some sanity checks
        if not all(len(v) == len(input_tokens) for v in out.values()):
            raise RuntimeError("Length mismatch")
        special_tokens = np.array([
            self.tokenizer.image_end_token_id,
            self.tokenizer.image_start_token_id,
            self.tokenizer.image_col_token_id,
            self.tokenizer.image_patch_token_id,
            self.tokenizer.image_low_res_token_id,
        ])[None, :]
        if np.any(target_tokens[mm_loss_masks != 0][:, None] == special_tokens):
            raise RuntimeError("A special token had a loss")
        if point_target_ids is not None:
            out["point_target_ids"] = point_target_ids
        if "point_target_ids" in out:
            if (out["point_target_ids"][:, 0] >= 0).sum() != (out["input_tokens"] == self.tokenizer.token_index_token_id).sum():
                raise RuntimeError("Message had mis-matched point tokens/token labels")
        return self._maybe_truncate_example(out)
