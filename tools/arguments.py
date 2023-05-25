import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.append("../src")
from vila.constants import *

tools_dir_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum input length."},
    )
    jitter_x: int = field(
        default=0,
        metadata={"help": "Range to jitter x during training"},
    )
    jitter_y: int = field(
        default=0,
        metadata={"help": "Range to jitter y during training"},
    )
    meta_model_type: str = field(
        default="ivila",
        metadata={"help": "The type of metamodel to use (e.g, `no_ivila`, `ivila`)"},
    )
    #################################
    ######### VILA Settings #########
    #################################

    added_special_sepration_token: str = field(
        default="SEP",
        metadata={
            "help": "The added special token for I-VILA models for separating the blocks/sentences/rows. Can be one of {SEP, BLK}. Default to `SEP`."
        },
    )
    textline_encoder_output: str = field(
        default="cls",
        metadata={
            "help": "How to obtain the group representation from the H-VILA model? Can be one of {cls, sep, average, last}. Default to `cls`."
        },
    )
    not_resume_training: bool = field(
        default=False,
        metadata={"help": "whether resume training from the existing checkpoints."},
    )

    fewshot_lr: float = field(
        default=1e-05,
        metadata={"help": "Learning rate for fewshot"},
    )

    num_fewshot_epochs: int = field(
        default=10,
        metadata={"help": "Num epochs for fewshot"},
    )

    def __post_init__(self):
        assert (self.added_special_sepration_token in ["BLK", "SEP"]) or (
            self.meta_model_type == "no_ivila"
        )

        if self.added_special_sepration_token == "BLK":
            self.added_special_sepration_token = "[BLK]"

        if self.added_special_sepration_token == "SEP":
            self.added_special_sepration_token = "[SEP]"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    do_retrain_finetune: bool = field(
        default=False,
        metadata={
            "help": "If true, does fewshot fine-tuning even if results file already exists."
        },
    )

    do_predict_before_fewshot_finetuning: bool = field(
        default=False, metadata={"help": "save predictions before fewshot finetuning"}
    )

    do_eval_predictions: bool = field(
        default=False, metadata={"help": "Save predictions for both test and eval"}
    )
    remove_bounding_boxes: bool = field(
        default=False, metadata={"help": "Remove bounding boxes from model inputs"}
    )

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    fewshot_episode_num: Optional[int] = field(
        default=-1,
        metadata={"help": "Which episode number to use for fewshot."},
    )

    data_dir: Optional[str] = field(
        default=os.path.join(tools_dir_path, "..", "data", "grotoap2_publisher_splits"),
        metadata={"help": "The directory with files containing the examples."},
    )
    
    load_from_huggingface: bool = field(
        default=True, metadata={"help": "Load datasets from huggingface hub"}
    )

    test_publisher_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the to journal to use for training (via the datasets library)."
        },
    )
    publisher_data_dict_file: Optional[str] = field(
        default=os.path.join(
            tools_dir_path,
            "..",
            "metadata",
            "grotoap2",
            "publisher_to_data_filename_dict_fewshot.json",
        ),
        metadata={
            "help": "The JSON file storing the (publisher:list of files corresponding to that publisher)"
        },
    )
    label_map_file: Optional[str] = field(
        default=os.path.join(
            tools_dir_path, "..", "metadata", "grotoap2", "remapped_label_map_file.json"
        ),
        metadata={"help": "The JSON file storing the (id:label_name)"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )
    excluded_pub_ids: str = field(
        default="2858749",
        metadata={"help": "Space separated list of PMC IDs of publications to ignore."},
    )
    excluded_pub_ids_layoutlmv: str = field(
        default="2858749 3035636",
        metadata={"help": "Space separated list of PMC IDs of publications to ignore."},
    )

    #################################
    ######### VILA Settings #########
    #################################

    agg_level: str = field(
        default="block",
        metadata={
            "help": "Used in some scenarios where the models will inject additional information to the models based on the agg_level"
        },
    )
    group_bbox_agg: str = field(
        default="first",
        metadata={
            "help": "The method to get the group bounding bbox, one of {union, first, center, last}. Default to `first`."
        },
    )
    max_line_per_page: Optional[int] = field(
        default=None,
        metadata={"help": "The number of textlines per page"},
    )
    max_tokens_per_line: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tokens per textline"},
    )
    max_block_per_page: Optional[int] = field(
        default=None,
        metadata={"help": "The number of block per page"},
    )
    max_tokens_per_block: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tokens per block"},
    )
    pred_file_name: Optional[str] = field(
        default="test_{test_dataset_name}_predictions.csv",
        metadata={"help": "The filename used for saving predictions."},
    )
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use for prediction."},
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()

        if self.dataset_name is not None:
            if self.dataset_name.lower() == "grotoap2":
                # fmt:off
                self.train_file      = "../data/grotoap2/train-token.json"
                self.validation_file = "../data/grotoap2/dev-token.json"
                self.test_file       = "../data/grotoap2/test-token.json"
                self.label_map_file  = "../data/grotoap2/labels.json"
                self.dataset_name = None
                # fmt:on

                self.max_line_per_page = MAX_LINE_PER_PAGE
                self.max_tokens_per_line = MAX_TOKENS_PER_LINE
                self.max_block_per_page = MAX_BLOCK_PER_PAGE
                self.max_tokens_per_block = MAX_TOKENS_PER_BLOCK
