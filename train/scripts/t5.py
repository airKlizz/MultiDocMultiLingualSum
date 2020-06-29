from scripts.summarization_trainer import SummarizationTrainer

from dataclasses import dataclass
from typing import Dict, List
from transformers import DataCollator
from transformers import T5ForConditionalGeneration, T5Tokenizer

import torch

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T5DataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


class T5SummarizationTrainer(SummarizationTrainer):

    data_collator = T5DataCollator()

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        model_cache_dir,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        summarize_prefix,
        wandb_project,
        wandb_run_name,
        version_column=None,
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
        self.tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=model_cache_dir,
        )
        self.summarize_prefix = summarize_prefix
        self.version_column = version_column
        self.summarize_prefixes = {
            "en": "summarize",
            "de": "zusammenfassen",
            "fr": "résume",
        }

    def format_text(self, example):
        # process the examples in input and target text format and the eos token at the end
        if self.version_column == None:
            example["input_text"] = "%s: %s </s>" % (
                self.summarize_prefix,
                example[self.document_column_name],
            )
        else:
            example["input_text"] = "%s: %s </s>" % (
                self.summarize_prefixes[example[self.version_column]],
                example[self.document_column_name],
            )
        example["target_text"] = "%s </s>" % example[self.summary_column_name]
        return example
