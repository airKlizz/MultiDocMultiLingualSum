import torch
import nlp
import dataclasses
import logging
import os
import sys
import wandb

import numpy as np
import torch

from dataclass_utils import *
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


class SummarizationTrainer(object):
    def __init__(
        self,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        wandb_project,
        wandb_run_name,
    ):
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length

        self.summary_column_name = summary_column_name
        self.document_column_name = document_column_name

        wandb.init(name=wandb_run_name, project=wandb_project, reinit=True)

    def cache_dataset(self, train_file_path, valid_file_path):
        torch.save(self.train_dataset, os.path.abspath(train_file_path))
        torch.save(self.valid_dataset, os.path.abspath(valid_file_path))

    def load_and_process_data(self, **load_dataset_kwargs):
        # load train and validation split of Multi-news
        self.train_dataset = nlp.load_dataset(split="train", **load_dataset_kwargs)
        self.valid_dataset = nlp.load_dataset(split="validation", **load_dataset_kwargs)

        # map format_text function to the dataset example wise
        self.train_dataset = self.train_dataset.map(
            self.format_text, load_from_cache_file=False
        )
        # map convert_to_features batch wise
        self.train_dataset = self.train_dataset.map(
            self.convert_to_features, batched=True, load_from_cache_file=False
        )

        self.valid_dataset = self.valid_dataset.map(
            self.format_text, load_from_cache_file=False
        )
        self.valid_dataset = self.valid_dataset.map(
            self.convert_to_features, batched=True, load_from_cache_file=False
        )

        # set the tensor type and the columns which the dataset should return
        columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
        self.train_dataset.set_format(type="torch", columns=columns)
        self.valid_dataset.set_format(type="torch", columns=columns)

    def format_text(self, example):
        example["input_text"] = example[self.document_column_name]
        example["target_text"] = example[self.summary_column_name]
        return example

    def convert_to_features(self, example_batch):
        # tokenize the examples
        input_encodings = self.tokenizer.batch_encode_plus(
            example_batch["input_text"],
            pad_to_max_length=True,
            max_length=self.input_max_length,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            example_batch["target_text"],
            pad_to_max_length=True,
            max_length=self.target_max_length,
        )

        encodings = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "target_ids": target_encodings["input_ids"],
            "target_attention_mask": target_encodings["attention_mask"],
        }

        return encodings

    @classmethod
    def train(cls, args_json_filename):
        parser = HfArgumentParser(
            (ClassArguments, DatasetArguments, DataTrainingArguments, TrainingArguments)
        )
        class_args, dataset_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(args_json_filename)
        )

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("Start training")
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed
        set_seed(training_args.seed)

        summarization_trainer = cls(**dataclasses.asdict(class_args))

        logger.info("Load and process dataset")
        summarization_trainer.load_and_process_data(**dataclasses.asdict(dataset_args))
        summarization_trainer.cache_dataset(
            data_args.train_file_path, data_args.valid_file_path
        )
        train_dataset = torch.load(data_args.train_file_path)
        valid_dataset = torch.load(data_args.valid_file_path)
        logger.info("Dataset ready")
        print(type(cls.data_collator.collate_batch))
        # Initialize our Trainer
        trainer = Trainer(
            model=summarization_trainer.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=cls.data_collator.collate_batch,
            prediction_loss_only=True,
        )

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=class_args.model_name_or_path
                if os.path.isdir(class_args.model_name_or_path)
                else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_master():
                summarization_trainer.tokenizer.save_pretrained(
                    training_args.output_dir
                )

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()

            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

            results.update(eval_output)

        return None
