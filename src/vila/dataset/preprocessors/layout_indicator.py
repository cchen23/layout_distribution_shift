import itertools
import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pysbd
from PIL import Image
from transformers import AutoProcessor

from ...utils import *
from .base import SimplePDFDataPreprocessor

Segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

preprocessors_dir_path = os.path.dirname(os.path.abspath(__file__))


def split_token_based_on_sentences_boundary(words: List[str]) -> List[Tuple[int, int]]:
    """
    Returns: List[Tuple(int, int)]
        a list of (start, end) for token indices within each sentence
    """

    if len(words) == 0:
        return [(0, 0)]
    combined_words = " ".join(words)

    char2token_mask = np.zeros(len(combined_words), dtype=np.int)

    acc_word_len = 0
    for idx, word in enumerate(words):
        word_len = len(word) + 1
        char2token_mask[acc_word_len : acc_word_len + word_len] = idx
        acc_word_len += word_len

    segmented_sentences = Segmenter.segment(combined_words)
    sent_boundary = [(ele.start, ele.end) for ele in segmented_sentences]

    split = []
    token_id_start = 0
    for start, end in sent_boundary:
        token_id_end = char2token_mask[start:end].max()
        if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
            token_id_end += 1  # (Including the end)
        split.append((token_id_start, token_id_end))
        token_id_start = token_id_end
    return split


class BaseLayoutIndicatorPDFDataPreprocessor(SimplePDFDataPreprocessor):
    def __init__(
        self,
        tokenizer,
        config,
        text_column_name="words",
        label_column_name="labels",
    ):
        super().__init__(tokenizer, config, text_column_name, label_column_name)

        self.added_special_sepration_token = config.added_special_sepration_token
        if self.added_special_sepration_token == "default":
            self.added_special_sepration_token = tokenizer.special_tokens_map[
                "sep_token"
            ]

    @abstractmethod
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:
        """It should be implemented differently for the functions"""

    def preprocess_sample(
        self,
        example: Dict,
        padding="max_length",
        max_length: int = 512,
        images_dir=os.path.join(preprocessors_dir_path, "../../../../data/grotoap2_images"),
    ) -> Dict:
        example, token_id_mapping_table = self.insert_layout_indicator(example)
        if self.processor:
            _, pub_id, page_num = example["files"].split("-")
            image_filepath = os.path.join(images_dir, f"{pub_id}-{page_num}.jpg")

            image = Image.open(image_filepath).convert("RGB")
            encoding = self.processor(
                image,
                example[self.text_column_name],
                boxes=example["bbox"],
                padding=padding,
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                max_length=max_length,
            )
        else:
            encoding = self.tokenizer(
                example[self.text_column_name],
                padding=padding,
                truncation=True,
                is_split_into_words=True,
                return_overflowing_tokens=True,
                max_length=max_length,
            )

        # original label and bbox from the input
        labels = example[self.label_column_name]
        bboxes = example["bbox"]

        # batched labels and bboxes
        batched_labels = []
        batched_bboxes = []
        previous_word_idx = None
        encoded_word_ids = []

        for batch_id in range(len(encoding["input_ids"])):
            word_ids = encoding.word_ids(batch_index=batch_id)

            cur_label_ids = []
            cur_bboxes = []

            for _i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    cur_label_ids.append(-100)
                    if (
                        encoding["input_ids"][batch_id][_i]
                        == self.special_tokens_map[
                            self.tokenizer.special_tokens_map["sep_token"]
                        ]
                    ):
                        cur_bboxes.append([1000, 1000, 1000, 1000])
                    else:
                        cur_bboxes.append([0, 0, 0, 0])

                elif word_idx != previous_word_idx:
                    cur_label_ids.append(int(labels[word_idx]))
                    cur_bboxes.append(bboxes[word_idx])

                else:
                    cur_label_ids.append(
                        int(labels[word_idx]) if self.config.label_all_tokens else -100
                    )
                    cur_bboxes.append(bboxes[word_idx])

                if not (_i == 0 and word_idx is None):
                    # Only updates the word_idx after the 0th item
                    # This is important because there would be cross-batch
                    # tokens.
                    previous_word_idx = word_idx

                if word_idx is not None:
                    if encoding["input_ids"][batch_id][_i] not in [
                        self.special_tokens_map[
                            self.tokenizer.special_tokens_map["sep_token"]
                        ],
                        self.special_tokens_map[self.added_special_sepration_token],
                    ]:
                        # Because we could possibly insert [SEP] or [BLK] tokens in
                        # this process.
                        encoded_word_ids.append(word_idx)

            batched_labels.append(cur_label_ids)
            batched_bboxes.append(cur_bboxes)

            # Find the last word id in this batch to handle
            # multi-batch samples
            for word_id in reversed(word_ids):
                if word_id is not None:
                    previous_word_idx = word_id
                    break

        new_id_to_original_id = {
            ele: idx for idx, ele in enumerate(token_id_mapping_table)
        }

        encoding["labels"] = batched_labels
        encoding["bbox"] = batched_bboxes
        encoding["encoded_word_ids"] = [
            new_id_to_original_id[ele] for ele in set(encoded_word_ids)
        ]

        return encoding


class BlockLayoutIndicatorPDFDataPreprocessor(BaseLayoutIndicatorPDFDataPreprocessor):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:
        processed_words = []
        processed_bbox = []
        processed_labels = []

        block_ids = example["block_ids"]
        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        pre_index = 0
        new_sequence_len = 0

        for block_id, gp in itertools.groupby(block_ids):
            cur_len = len(list(gp))
            token_id_mapping_table[pre_index : pre_index + cur_len] = list(
                range(new_sequence_len, new_sequence_len + cur_len)
            )
            processed_words.extend(
                words[pre_index : pre_index + cur_len]
                + [self.added_special_sepration_token]
            )
            processed_bbox.extend(
                bbox[pre_index : pre_index + cur_len]
                + [union_box(bbox[pre_index : pre_index + cur_len])]
            )
            processed_labels.extend(labels[pre_index : pre_index + cur_len] + [-100])
            pre_index += cur_len
            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
            "files": example["files"],
        }, token_id_mapping_table


class RowLayoutIndicatorPDFDataPreprocessor(BaseLayoutIndicatorPDFDataPreprocessor):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:
        processed_words = []
        processed_bbox = []
        processed_labels = []

        line_ids = example["line_ids"]  # Changed
        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        pre_index = 0
        new_sequence_len = 0

        for line_id, gp in itertools.groupby(line_ids):  # Changed
            cur_len = len(list(gp))
            token_id_mapping_table[pre_index : pre_index + cur_len] = list(
                range(new_sequence_len, new_sequence_len + cur_len)
            )
            processed_words.extend(
                words[pre_index : pre_index + cur_len]
                + [self.added_special_sepration_token]
            )
            processed_bbox.extend(
                bbox[pre_index : pre_index + cur_len]
                + [union_box(bbox[pre_index : pre_index + cur_len])]
            )
            processed_labels.extend(labels[pre_index : pre_index + cur_len] + [-100])
            pre_index += cur_len
            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
        }, token_id_mapping_table


class SentenceLayoutIndicatorPDFDataPreprocessor(
    BaseLayoutIndicatorPDFDataPreprocessor
):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:
        processed_words = []
        processed_bbox = []
        processed_labels = []

        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        token_splits = split_token_based_on_sentences_boundary(words)

        new_sequence_len = 0
        for start, end in token_splits:
            token_id_mapping_table[start:end] = list(
                range(new_sequence_len, new_sequence_len + end - start)
            )
            processed_words.extend(
                words[start:end] + [self.added_special_sepration_token]
            )
            processed_bbox.extend(bbox[start:end] + [union_box(bbox[start:end])])
            processed_labels.extend(labels[start:end] + [-100])

            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
        }, token_id_mapping_table
