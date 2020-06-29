from nltk.tokenize import sent_tokenize
from nlp import load_metric
import numpy as np
from tqdm import tqdm

from baselines.baseline import Baseline


class RougeOracle(Baseline):
    def __init__(self, name, rouge_type="rouge2", rouge_method="precision"):
        super().__init__(name)
        self.rouge_metric = load_metric("rouge")
        self.rouge_type = rouge_type
        if rouge_method == "precision":
            self.rouge_method = 0
        elif rouge_method == "recall":
            self.rouge_method = 1
        elif rouge_method == "fmeasure":
            self.rouge_method = 2
        else:
            raise ValueError('rouge_method must be "precision", "recall" or "fmeasure"')

    def _calculate_rouge(self, prediction, reference):
        score = self.rouge_metric.compute(
            [prediction],
            [reference],
            rouge_types=[self.rouge_type],
            use_agregator=False,
        )
        value = score[self.rouge_type][0][self.rouge_method]
        return value

    def rank_sentences(
        self, dataset, document_column_name, run_summary_colunm_name, **kwargs
    ):
        def run_rouge_oracle(example):
            sentences = sent_tokenize(example[document_column_name])
            reference = example[run_summary_colunm_name]
            scores = [self._calculate_rouge(sent, reference) for sent in sentences]

            example[self.name] = {
                "sentences": sentences,
                "scores": scores,
            }
            return example

        dataset = dataset.map(run_rouge_oracle)
        return dataset
