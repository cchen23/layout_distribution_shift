# Modified based on https://github.com/huggingface/transformers/tree/master/examples/token-classification
import copy
import glob
import logging
import os

import sys
import itertools

import json
import numpy as np
from datasets import ClassLabel, concatenate_datasets, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from utils import (
    get_best_checkpoint,
    load_json,
    classification_report,
    save_predictions,
    DataCollatorForTokenClassification,
)

from arguments import ModelArguments, DataTrainingArguments

sys.path.append("../src")
from vila.dataset.preprocessors import instantiate_dataset_preprocessor

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        filename="./log.txt",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: only accepts json files for now.
    publisher_to_data_names_dict = load_json(data_args.publisher_data_dict_file)[
        data_args.test_publisher_name
    ]
    train_data_names = publisher_to_data_names_dict["train_filenames"]
    dev_data_names = publisher_to_data_names_dict["dev_filenames"]

    test_split_names = [
        split_name
        for split_name in publisher_to_data_names_dict
        if split_name[:5] == "test_"
    ]
    test_data_names_dict = dict()
    for split_name in test_split_names:
        test_data_names_dict[split_name] = publisher_to_data_names_dict[split_name]
    if "fewshot_finetuning_sets_names_dict" in publisher_to_data_names_dict:
        fewshot_finetuning_sets_names_dict = publisher_to_data_names_dict[
            "fewshot_finetuning_sets_names_dict"
        ]
        if data_args.fewshot_episode_num >= 0:
            for (
                fewshot_finetuning_set_name,
                fewshot_finetuning_set_dict,
            ) in fewshot_finetuning_sets_names_dict.items():
                fewshot_finetuning_set_dict["train"] = [
                    filename.format(episode_num=data_args.fewshot_episode_num)
                    for filename in fewshot_finetuning_set_dict["train"]
                ]
                fewshot_finetuning_set_dict["dev"] = [
                    filename.format(episode_num=data_args.fewshot_episode_num)
                    for filename in fewshot_finetuning_set_dict["dev"]
                ]
    else:
        fewshot_finetuning_sets_names_dict = dict()
        fewshot_finetuning_train_data_names = [
            filename.format(episode_num=data_args.fewshot_episode_num)
            for filename in publisher_to_data_names_dict[
                "fewshot_finetuning_train_filenames_by_episode"
            ]
        ]
        fewshot_finetuning_dev_data_names = [
            filename.format(episode_num=data_args.fewshot_episode_num)
            for filename in publisher_to_data_names_dict[
                "fewshot_finetuning_dev_filenames_by_episode"
            ]
        ]
        fewshot_finetuning_sets_names_dict["single"] = {
            "train": fewshot_finetuning_train_data_names,
            "dev": fewshot_finetuning_dev_data_names,
        }

    all_data_files = (
        train_data_names
        + dev_data_names
        + list(itertools.chain(*test_data_names_dict.values()))
    )
    for (
        fewshot_finetuning_set_name,
        fewshot_finetuning_set_filenames,
    ) in fewshot_finetuning_sets_names_dict.items():
        all_data_files += fewshot_finetuning_set_filenames["train"]
        all_data_files += fewshot_finetuning_set_filenames["dev"]

    data_files = {}
    for data_filename in all_data_files:
        data_files[os.path.split(data_filename)[-1]] = os.path.join(
            data_args.data_dir, data_filename
        )
    extension = "json"  # NOTE: Hard-coded.
    print(f"Using data files {data_files.keys()}")
    datasets = load_dataset(extension, data_files=data_files, field="data")
    for data_file_key, data_file_value in data_files.items():
        with open(data_file_value, "r") as f:
            files_list = json.load(f)["files"]
        datasets[data_file_key] = datasets[data_file_key].add_column(
            "files", files_list
        )

    def filter_fn(example_dict):
        file_id = example_dict["files"]
        if "layoutlmv" in model_args.model_name_or_path:
            return (
                sum(
                    [
                        excluded_pub_id in file_id
                        for excluded_pub_id in data_args.excluded_pub_ids_layoutlmv.split()
                    ]
                )
                == 0
            )
        else:
            return (
                sum(
                    [
                        excluded_pub_id in file_id
                        for excluded_pub_id in data_args.excluded_pub_ids.split()
                    ]
                )
                == 0
            )

    for dataset_file_key, dataset in datasets.items():
        datasets[dataset_file_key] = dataset.filter(filter_fn)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Create train, test, validation splits.
    datasets["train"] = concatenate_datasets(
        [datasets[os.path.split(data_name)[-1]] for data_name in train_data_names]
    )
    datasets["dev"] = concatenate_datasets(
        [datasets[os.path.split(data_name)[-1]] for data_name in dev_data_names]
    )
    for split_name in test_split_names:
        datasets[split_name] = concatenate_datasets(
            [
                datasets[os.path.split(data_name)[-1]]
                for data_name in publisher_to_data_names_dict[split_name]
            ]
        )
    for (
        fewshot_finetuning_set_name,
        fewshot_finetuning_set_filenames,
    ) in fewshot_finetuning_sets_names_dict.items():
        datasets[
            f"fewshot_finetuning_train_{fewshot_finetuning_set_name}"
        ] = concatenate_datasets(
            [
                datasets[os.path.split(data_name)[-1]]
                for data_name in fewshot_finetuning_set_filenames["train"]
            ]
        )
        datasets[
            f"fewshot_finetuning_dev_{fewshot_finetuning_set_name}"
        ] = concatenate_datasets(
            [
                datasets[os.path.split(data_name)[-1]]
                for data_name in fewshot_finetuning_set_filenames["dev"]
            ]
        )

    for split_name in datasets:
        datasets[split_name] = datasets[split_name].shuffle(seed=training_args.seed)

    features = datasets["train"].features
    label_column_name = "labels"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if data_args.label_map_file is not None:
        label_list = list(load_json(data_args.label_map_file).values())
    else:
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
        else:
            label_list = get_label_list(datasets["train"][label_column_name])
            label_list = [str(ele) for ele in label_list]  # Ensure the ele is a string

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_name_or_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.meta_model_type == "ivila":
        if (
            model_args.added_special_sepration_token
            not in tokenizer.special_tokens_map.values()
        ):
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        model_args.added_special_sepration_token
                    ]
                }
            )
            model.resize_token_embeddings(len(tokenizer))

        logger.info(f"The used agg level is {data_args.agg_level}")
        data_args.added_special_sepration_token = (
            model_args.added_special_sepration_token
        )
        preprocessor = instantiate_dataset_preprocessor(
            "layout_indicator", tokenizer, data_args
        )
    elif model_args.meta_model_type == "no_ivila":
        preprocessor = instantiate_dataset_preprocessor("base", tokenizer, data_args)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": model_args.jitter_x,
                "jitter_y": model_args.jitter_y,
            },
        )
        if data_args.remove_bounding_boxes:
            train_dataset = train_dataset.remove_columns(["bbox"])
        print(f"Train dataset size {len(train_dataset)}")

    if training_args.do_eval or data_args.do_eval_predictions:
        eval_dataset = datasets["dev"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        else:
            assert "For now, need max_val_samples and max_train_samples"
        eval_dataset_for_predictions = eval_dataset.map(
            preprocessor.preprocess_sample,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[],
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": 0,
                "jitter_y": 0,
            },  # No jitter for eval or test.
        )
        eval_dataset_batched = eval_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": 0,
                "jitter_y": 0,
            },  # No jitter for eval or test.
        )
        if data_args.remove_bounding_boxes:
            eval_dataset_for_predictions = eval_dataset.remove_columns(["bbox"])
            eval_dataset_batched = eval_dataset_batched.remove_columns(["bbox"])
        print(f"Eval dataset size {len(eval_dataset_batched)}")

    for split_name in test_split_names:
        split_test_dataset = datasets[split_name]
        if data_args.max_test_samples is not None:
            split_test_dataset = split_test_dataset.select(
                range(data_args.max_test_samples)
            )
        split_test_dataset = split_test_dataset.map(
            preprocessor.preprocess_sample,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": 0,
                "jitter_y": 0,
            },  # No jitter for eval or test.
        )
        if data_args.remove_bounding_boxes:
            split_test_dataset = split_test_dataset.remove_columns(["bbox"])
        datasets[split_name] = split_test_dataset
        print(f"Test dataset {split_name} size {len(split_test_dataset)}")

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        max_length=model_args.max_length,
    )

    def compute_metrics(p, label_list):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_predictions = list(itertools.chain.from_iterable(true_predictions))
        true_labels = list(itertools.chain.from_iterable(true_labels))
        results = classification_report(y_true=true_labels, y_pred=true_predictions)
        results.pop("detailed")
        return results

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset_batched,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, label_list),
    )

    # Training
    if training_args.do_train:
        if not model_args.not_resume_training:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_name_or_path):
                checkpoint = model_name_or_path
            else:
                checkpoint = None
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    checkpoint = get_best_checkpoint(training_args.output_dir)
    trainer.model = trainer.model.from_pretrained(checkpoint).to(training_args.device)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(eval_dataset_batched)
        )
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset_batched))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    test_label_list = label_list
    test_num_labels = num_labels

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        print("*** Predict ***")
        if test_num_labels == num_labels:
            datasets_list = [datasets[split_name] for split_name in test_split_names]
            dataset_names_list = copy.deepcopy(test_split_names)
        else:
            datasets_list = []
            dataset_names_list = []
        if data_args.do_eval_predictions:
            datasets_list.append(eval_dataset_for_predictions)
            dataset_names_list.append(f"eval_{data_args.test_publisher_name}")
        for dataset, dataset_name in zip(datasets_list, dataset_names_list):
            output_test_predictions_file = os.path.join(
                training_args.output_dir,
                data_args.pred_file_name.format(test_dataset_name=dataset_name),
            )
            logger.info(
                f"The test file will be saved to {output_test_predictions_file}"
            )
            print(f"The test file will be saved to {output_test_predictions_file}")

            save_predictions(dataset, trainer.model, output_test_predictions_file)

    # Do few-shot fine-tuning.
    for (
        fewshot_finetuning_set_name,
        fewshot_finetuning_set_filenames,
    ) in fewshot_finetuning_sets_names_dict.items():
        fewshot_finetuning_train_dataset = datasets[
            f"fewshot_finetuning_train_{fewshot_finetuning_set_name}"
        ]
        fewshot_finetuning_eval_dataset = datasets[
            f"fewshot_finetuning_dev_{fewshot_finetuning_set_name}"
        ]
        fewshot_finetuning_train_dataset = fewshot_finetuning_train_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=fewshot_finetuning_train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": model_args.jitter_x,
                "jitter_y": model_args.jitter_y,
            },
        )

        fewshot_finetuning_eval_dataset = fewshot_finetuning_eval_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=fewshot_finetuning_eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "max_length": model_args.max_length,
                "jitter_x": model_args.jitter_x,
                "jitter_y": model_args.jitter_y,
            },
        )
        if data_args.fewshot_episode_num >= 0:
            fewshot_output_dir = os.path.join(
                training_args.output_dir,
                f"fewshot_{fewshot_finetuning_set_name}_episode_{data_args.fewshot_episode_num}_lr_{model_args.fewshot_lr}/",
            )
        else:
            fewshot_output_dir = os.path.join(
                training_args.output_dir,
                f"fewshot_{fewshot_finetuning_set_name}_lr_{model_args.fewshot_lr}/",
            )
        print("fewshot_output_dir", fewshot_output_dir)

        fewshot_training_args = TrainingArguments(
            output_dir=fewshot_output_dir,
            do_train=True,
            evaluation_strategy=training_args.evaluation_strategy,
            learning_rate=model_args.fewshot_lr,
            num_train_epochs=model_args.num_fewshot_epochs,
            warmup_steps=0,
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
        )

        # Do not re-train.
        output_test_predictions_file = os.path.join(
            fewshot_training_args.output_dir,
            data_args.pred_file_name.format(
                test_dataset_name=f"test_filenames_{fewshot_finetuning_set_name}"
            ),
        )
        if (
            os.path.exists(output_test_predictions_file)
            and not data_args.do_retrain_finetune
        ):
            print(
                f"{output_test_predictions_file} already exists! skipping fine-tuning"
            )
            continue

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_name_or_path,
            num_labels=test_num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if model_args.meta_model_type == "ivila":
            model.resize_token_embeddings(len(tokenizer))

        model = model.from_pretrained(
            checkpoint, config=config, ignore_mismatched_sizes=True
        )
        fewshot_trainer = Trainer(
            model=model,
            args=fewshot_training_args,
            train_dataset=fewshot_finetuning_train_dataset,
            eval_dataset=fewshot_finetuning_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: compute_metrics(x, test_label_list),
        )
        print("warmup steps", fewshot_trainer.args.warmup_steps)

        if data_args.do_predict_before_fewshot_finetuning:
            logger.info("*** Predict before few-shot fine-tuning ***")
            print("*** Predict before few-shot fine-tuning ***")
            datasets_list = [datasets[split_name] for split_name in test_split_names]
            dataset_names_list = copy.deepcopy(test_split_names)
            if data_args.do_eval_predictions:
                datasets_list.append(eval_dataset_for_predictions)
                dataset_names_list.append(f"eval_{data_args.test_publisher_name}")
            for dataset, dataset_name in zip(datasets_list, dataset_names_list):
                output_test_predictions_file = os.path.join(
                    fewshot_training_args.output_dir,
                    data_args.pred_file_name.format(
                        test_dataset_name=f"{dataset_name}_before_fewshot"
                    ),
                )
                logger.info(
                    f"The test file will be saved to {output_test_predictions_file}"
                )
                print(f"The test file will be saved to {output_test_predictions_file}")
                save_predictions(
                    dataset, fewshot_trainer.model, output_test_predictions_file
                )

        # Few-shot training
        print(f"starting fewshot training from checkpoint {checkpoint}")
        train_result = fewshot_trainer.train()
        metrics = train_result.metrics
        fewshot_trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(fewshot_finetuning_train_dataset)
        )
        metrics["train_samples_fewshot"] = min(
            max_train_samples, len(fewshot_finetuning_train_dataset)
        )

        fewshot_trainer.log_metrics(
            f"train_fewshot_{fewshot_finetuning_set_name}", metrics
        )
        fewshot_trainer.save_metrics(
            f"train_fewshot_{fewshot_finetuning_set_name}", metrics
        )
        fewshot_trainer.save_state()

        fewshot_checkpoint = get_best_checkpoint(fewshot_training_args.output_dir)
        print(f"loading fewshot training checkpoint {fewshot_checkpoint}")
        fewshot_trainer.model = fewshot_trainer.model.from_pretrained(
            fewshot_checkpoint
        ).to(training_args.device)

        # Test again.
        logger.info("*** Predict after few-shot fine-tuning ***")
        print("*** Predict after few-shot fine-tuning ***")
        datasets_list = [datasets[split_name] for split_name in test_split_names]
        dataset_names_list = copy.deepcopy(test_split_names)
        if data_args.do_eval_predictions:
            datasets_list.append(eval_dataset_for_predictions)
            dataset_names_list.append(f"eval_{data_args.test_publisher_name}")
        for dataset, dataset_name in zip(datasets_list, dataset_names_list):
            output_test_predictions_file = os.path.join(
                fewshot_training_args.output_dir,
                data_args.pred_file_name.format(test_dataset_name=dataset_name),
            )
            logger.info(
                f"The test file will be saved to {output_test_predictions_file}"
            )
            print(f"The test file will be saved to {output_test_predictions_file}")

            save_predictions(
                dataset, fewshot_trainer.model, output_test_predictions_file
            )

        # Delete fewshot checkpoints.
        fewshot_files_to_delete = [
            f
            for f in glob.glob(os.path.join(fewshot_training_args.output_dir, "*"))
            if ".json" not in f and ".csv" not in f
        ]
        for filepath in fewshot_files_to_delete:
            os.system(f"rm -r {filepath}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
