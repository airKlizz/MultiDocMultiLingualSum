from scripts.summarization_trainer import SummarizationTrainer

from dataclasses import dataclass
from typing import Dict, List
from transformers import DataCollator
from transformers import BartForConditionalGeneration, BartTokenizer

import torch
from torch import nn

@dataclass
class BartDataCollator(DataCollator):
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


class DistilbartSummarizationTrainer(SummarizationTrainer):

    data_collator = BartDataCollator()

    def __init__(
        self,
        model_name_or_path, # teacher
        tokenizer_name,
        model_cache_dir,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        wandb_project,
        wandb_run_name,
        student_encoder_layers,
        student_decoder_layers,
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
        teacher = BartForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=model_cache_dir,
        ).eval()

        student_updates = {
            "decoder_layers": student_decoder_layers,
            "encoder_layers": student_encoder_layers,
        }
        d_layers_to_copy = self._get_layers_to_copy(student_updates["decoder_layers"], teacher.config.decoder_layers)
        e_layers_to_copy: List = self._get_layers_to_copy(student_updates["encoder_layers"], teacher.config.encoder_layers)
        kw = teacher.config.to_diff_dict()
        kw.update(student_updates)
        # Copy weights
        student_cfg = BartConfig(**kw)
        student = BartForConditionalGeneration(student_cfg)
        student, _ = self._init_student(student, teacher)
        self._copy_to_student(d_layers_to_copy, e_layers_to_copy, student_encoder_layers, student_decoder_layers, student, teacher)
        self.model = student
    
    def _init_student(self, student, teacher):
        teacher_state_dict = teacher.state_dict()
        info = student.load_state_dict(teacher_state_dict, strict=False)
        assert info.missing_keys == [], info.missing_keys
        return student, info

    def _copy_to_student(self, d_layers_to_copy, e_layers_to_copy, student_encoder_layers, student_decoder_layers, student, teacher):
        if teacher.config.model_type == "t5":
            assert ValueError("T5 not implemented")
        self.different_encoder = student_encoder_layers != teacher.config.encoder_layers
        self.different_decoder = student_decoder_layers != teacher.config.decoder_layers
        if self.different_decoder:
            self._copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
        if self.different_encoder:
            self._copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)

    def _copy_layers(self, teacher_layers: nn.ModuleList, student_layers: nn.ModuleList, layers_to_copy: List) -> None:
        layers_to_copy = nn.ModuleList([l for i, l in enumerate(teacher_layers) if i in layers_to_copy])
        assert len(student_layers) == len(layers_to_copy), f"{len(student_layers)} != {len(layers_to_copy)}"
        student_layers.load_state_dict(layers_to_copy.state_dict())

    def _get_layers_to_copy(self, n_to_get, tot):
        all_layers = list(range(tot))
        if tot == 12:  # Alternating for special cases
            layers_to_copy = {  # maps  num layers in student -> which teacher layers to copy
                1: [0],
                2: [0, 6],
                3: [0, 6, 11],
                4: [0, 4, 8, 11],
                6: [0, 2, 4, 7, 9, 11],
                9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
                12: all_layers,
            }
            return layers_to_copy[n_to_get]
        else:
            return all_layers[:n_to_get]  # TODO: better version on theseus-bart branch