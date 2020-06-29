import numpy as np
from tqdm import tqdm

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from baselines.baseline import Baseline


class TextRankSumy(Baseline):

    """ Description from https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd
    TextRank is an algorithm based on PageRank, which often used in keyword extraction and text summarization.
    """

    """ Implementation
    Wrapper of https://github.com/miso-belica/sumy
    """

    def __init__(self, name, language):
        super().__init__(name)
        self.language = language
        self.summarizer = TextRankSummarizer()
        self.language = language

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        all_sentences = []
        all_scores = []
        for document in tqdm(dataset[document_column_name]):
            sentences, scores = self.run_single(document)
            all_sentences.append(sentences)
            all_scores.append(scores)

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)

    def run_single(self, document):
        parser = PlaintextParser.from_string(document, Tokenizer(self.language))
        document = parser.document

        self.summarizer._ensure_dependencies_installed()

        ratings = self.summarizer.rate_sentences(document)

        sentences, scores = zip(*ratings.items())

        return list(map(str, document.sentences)), list(scores)
