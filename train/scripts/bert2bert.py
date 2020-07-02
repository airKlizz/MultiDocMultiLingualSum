from scripts.summarization_trainer import SummarizationTrainer

from dataclasses import dataclass
from typing import Dict, List
from transformers import DataCollator
from transformers import EncoderDecoderModel, BertTokenizer

import torch


@dataclass
class Bert2BertDataCollator(object):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        decoder_input_ids = torch.stack([example["target_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


class Bert2BertSummarizationTrainer(SummarizationTrainer):

    data_collator = Bert2BertDataCollator()

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
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_name_or_path, model_name_or_path, cache_dir=model_cache_dir,
        )
