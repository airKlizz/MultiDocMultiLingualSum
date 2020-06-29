from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
import numpy as np

from baselines.baseline import Baseline

from baselines import baselines


class Combine(Baseline):

    """ Description 
    Class which can combine an extractive and abstractive baselines by performing the abstractive baseline on sentences ordered by the extractive one.
    """

    def __init__(
        self,
        name,
        extractive_class,
        abstractive_class,
        extractive_args,
        abstractive_args,
    ):
        super().__init__(name)
        self.extractive = baselines.use(extractive_class, **extractive_args)
        self.abstractive = baselines.use(abstractive_class, **abstractive_args)

    def get_summaries(
        self, dataset, document_column_name, extractive_args, abstractive_args
    ):
        # Extractive step
        print(f"Run extractive {self.extractive.name}")
        dataset = self.extractive.rank_sentences(
            dataset, document_column_name, **extractive_args
        )

        # Truncate best sentences to the input length of the model
        # and re-order sentences based on their original order in the document
        def truncate_and_order_example(example):
            scores = np.array(example[self.extractive.name]["scores"])
            sentences = example[self.extractive.name]["sentences"]
            sorted_ix = np.argsort(scores)[::-1]
            ranked_sentences = [sentences[j] for j in sorted_ix]
            truncated_and_ordered_sentences = self._truncate_and_order(
                sentences, ranked_sentences, self.abstractive.input_max_length
            )
            example["abstractive_input"] = " ".join(truncated_and_ordered_sentences)
            return example

        dataset = dataset.map(truncate_and_order_example)

        # Abstractive step
        print(f"Run abstractive {self.abstractive.name}")
        self.abstractive.name = self.name
        dataset = self.abstractive.get_summaries(
            dataset, "abstractive_input", **abstractive_args
        )

        return dataset

    def _truncate_and_order(
        self, original_sentences, ranked_sentences, input_max_length
    ):
        truncated_sentences = []
        i = 0
        while self._num_words(
            " ".join(truncated_sentences)
        ) < input_max_length and i < len(ranked_sentences):
            truncated_sentences.append(ranked_sentences[i])
            i += 1

        truncated_and_ordered_sentences = []
        for sentence in original_sentences:
            if sentence in truncated_sentences:
                truncated_and_ordered_sentences.append(sentence)

        return truncated_and_ordered_sentences

    def _num_words(self, text):
        text = text.translate(str.maketrans(punctuation, " " * len(punctuation)))
        return len(text.split(" "))
