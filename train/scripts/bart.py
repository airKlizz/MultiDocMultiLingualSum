from scripts.summarization_trainer import SummarizationTrainer

from dataclasses import dataclass
from typing import Dict, List
from transformers import DataCollator
from transformers import BartForConditionalGeneration, BartTokenizer

import torch


@dataclass
class BartDataCollator():
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        y = torch.stack([example["target_ids"] for example in batch])
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == 1] = -100
        lm_labels[lm_labels[:, :] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": y_ids,
        }


class BartSummarizationTrainer(SummarizationTrainer):

    data_collator = BartDataCollator()

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        model_cache_dir,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        wandb_project,
        wandb_run_name,
        **kwargs,
    ):
        super().__init__(
            input_max_length,
            target_max_length,
            summary_column_name,
            document_column_name,
            wandb_project,
            wandb_run_name,
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=model_cache_dir,
        )
