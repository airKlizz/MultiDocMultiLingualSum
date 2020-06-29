import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from math import log

from baselines.baseline import Baseline


class TFIDF(Baseline):

    """ Description
    Sentences are scored with their similarity with the title of the article. 
    """

    """ Implementation
    It follows the paper: https://arxiv.org/pdf/1801.10198.pdf
    """

    def __init__(self, name, language):
        super().__init__(name)
        self.stop_words = set(stopwords.words(language))

    def rank_sentences(
        self, dataset, document_column_name, title_column_name, **kwargs,
    ):
        all_sentences = []
        all_scores = []
        for title, document in tqdm(
            zip(dataset[title_column_name], dataset[document_column_name])
        ):
            sentences, scores = self.run_single(title, document)
            all_sentences.append(sentences)
            all_scores.append(scores)

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)

    def run_single(self, title, document):
        sentences = sent_tokenize(document)
        title_words = [w for w in word_tokenize(title) if not w in self.stop_words]

        scores = [0] * len(sentences)
        sentences_words = list(map(word_tokenize, sentences))
        n_d = len(sentences)
        for title_word in title_words:
            n_dw = self._compute_n_dw(title_word, sentences_words)
            if n_dw == 0:
                continue
            for i, sentence_words in enumerate(sentences_words):
                n_w = sentence_words.count(title_word)
                scores[i] += n_w * log(n_d / n_dw)

        return sentences, scores

    # Return the number of sentences containing the word
    def _compute_n_dw(self, word, sentences_words):
        count = 0
        for sentence_words in sentences_words:
            if word in sentence_words:
                count += 1
        return count
