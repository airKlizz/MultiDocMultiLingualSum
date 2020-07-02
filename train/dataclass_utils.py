import torch

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from nlp.splits import Split


@dataclass
class ClassArguments:
    """
    Arguments to init the SummarizationTrainer class.
    """

    input_max_length: int = field(metadata={"help": "Max tokens of the input"})
    target_max_length: int = field(metadata={"help": "Max tokens of the targer"})

    summary_column_name: str = field(metadata={"help": "Name of the summary column"})
    document_column_name: str = field(metadata={"help": "Name of the document column"})

    wandb_project: str = field(metadata={"help": "Name of the wandb project"})
    wandb_run_name: str = field(metadata={"help": "Name of the wandb run"})

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    title_column_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the title column"}
    )
    summarize_prefix: Optional[str] = field(
        default=None, metadata={"help": "Prefix for T5"}
    )
    title_prefix: Optional[str] = field(
        default=None, metadata={"help": "Title prefix for T5"}
    )
    version_column: Optional[str] = field(
        default=None, metadata={"help": "Name of the version column"}
    )
    student_encoder_layers: Optional[int] = field(
        default=12, metadata={"help": "Number of encoder layer for Distilbart"}
    )
    student_decoder_layers: Optional[int] = field(
        default=6, metadata={"help": "Number of decoder layer for Distilbart"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: Optional[str] = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        metadata={"help": "Max input length for the target text"},
    )


@dataclass
class DatasetArguments:
    """
    Arguments to load the dataset.
    """

    path: str = field(
        metadata={
            "help": "Path to the dataset processing script with the dataset builder"
        },
    )
    ignore_verifications: bool = field(default=None)
    save_infos: bool = field(default=None)
    name: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
