""" Baseline base class."""

import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import load_metric
import pyarrow as pa


class Baseline(object):
    def __init__(self, name):
        """ 
        A Baseline is the base class for all baselines.
        """
        self.name = name.replace(" ", "-").lower()
        np.random.seed(42)

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        """
        Run the extractive baseline for all documents by associating a score to each sentences.
        Args:
            dataset (datasets.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
        Return:
            dataset (datasets.Dataset): dataset with a new column containing sentences and scores.
        """

        raise NotImplementedError()

    def get_summaries(
        self,
        dataset,
        document_column_name,
        num_sentences,
        non_redundant=False,
        rouge_type="rouge2",
        rouge_method="recall",
        redundance_threshold=0.05,
        ordering=False,
        **kwargs,
    ):
        """
        Get the summary of each documents.
        Args:
            dataset (datasets.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            num_sentences (int or list): number of sentences in the summaries or list of number of sentences in the summaries
            non_redundant (bool): apply non redundant 
            rouge_type (str): type of rouge to use if non_redundant
            rouge_method (str): method of rouge to use if non_redundant
            redundance_threshold (float): redundance threshold if non_redundant
            ordering (bool): apply ordering
            **kwargs: arguments to pass to the run function
        Return:
            dataset (datasets.Dataset): dataset with a new column for hypothesis
        """
        dataset = self.rank_sentences(dataset, document_column_name, **kwargs)

        if non_redundant:
            self._init_rouge(rouge_type, rouge_method)

        if isinstance(num_sentences, int) or num_sentences == None:
            num_sentences = [num_sentences for i in range(len(dataset))]
        if len(num_sentences) != len(dataset):
            raise ValueError("documents and num_sentences must have the same length")

        dataset = Baseline.append_column(dataset, num_sentences, "num_sentences")

        def get_extractive_summary(example):
            scores = np.array(example[self.name]["scores"])
            sentences = example[self.name]["sentences"]
            sorted_ix = np.argsort(scores)[::-1]

            if non_redundant:
                sorted_ix_non_redundant = []
                redundance_score = 0
                for k in sorted_ix:
                    redundance_score = self._calculate_rouge(
                        sentences[k],
                        " ".join([sentences[i] for i in sorted_ix_non_redundant]),
                    )
                    if redundance_score < redundance_threshold:
                        sorted_ix_non_redundant.append(k)
                    if len(sorted_ix_non_redundant) >= example["num_sentences"]:
                        break
                sorted_ix = sorted_ix_non_redundant

            if ordering:
                summary_sentences = [
                    sentences[j] for j in sorted_ix[: example["num_sentences"]]
                ]
                hyp = ""
                for sentence in sentences:
                    if sentence in summary_sentences:
                        hyp += sentence + " "

            else:
                hyp = " ".join(
                    [sentences[j] for j in sorted_ix[: example["num_sentences"]]]
                )

            example[f"{self.name}_hypothesis"] = hyp
            return example

        dataset = dataset.map(get_extractive_summary)
        dataset.drop("num_sentences")
        return dataset

    def compute_rouge(
        self,
        dataset,
        document_column_name,
        summary_colunm_name,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        **kwargs,
    ):
        """
        Generate hypotheses and compute ROUGE score between summaries and hypotheses
        Args:
            dataset (datasets.Dataset): dataset containing document to summarize
            document_column_name (str): name of the column of the dataset containing documents
            summary_colunm_name (str): name of the column of the dataset containing summaries
            rouge_types (lst(str)): list of ROUGE types you want to compute
            **kwargs: arguments to pass to the run function
        Return:
            score (dict(Score)): dict of ROUGE types with the score (see datasets metrics for details)
        """

        dataset = self.get_summaries(dataset, document_column_name, **kwargs)

        rouge_metric = load_metric("rouge")

        def compute_rouge_batch(example):
            predictions = example[f"{self.name}_hypothesis"]
            references = example[summary_colunm_name]
            rouge_metric.add_batch(predictions, references)

        dataset.map(compute_rouge_batch, batched=True)
        return dataset, rouge_metric.compute(rouge_types=rouge_types)

    def _init_rouge(self, rouge_type, rouge_method):
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

    @staticmethod
    def append_column(dataset, data, column_name):
        data = pa.array(data)
        dataset._data = dataset.data.append_column(column_name, data)
        return dataset
