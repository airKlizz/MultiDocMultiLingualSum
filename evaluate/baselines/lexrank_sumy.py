import numpy as np
from tqdm import tqdm

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from baselines.baseline import Baseline


class LexRankSumy(Baseline):

    """ Description from https://arxiv.org/pdf/1905.13164.pdf
    LexRank (Erkan and Radev, 2004) is a widely-used graph-based extractive summarizer; 
    we build a graph with paragraphs as nodes andedges weighted by tf-idf cosine similarity; 
    we run a PageRank-like algorithm on this graph to rank and select paragraphs until 
    the length of the ground-truth summary is reached.
    """

    """ Implementation
    Wrapper of https://github.com/miso-belica/sumy
    """

    def __init__(self, name, language):
        super().__init__(name)
        self.language = language
        self.summarizer = LexRankSummarizer()

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

        sentences_words = [self.summarizer._to_words_set(s) for s in document.sentences]
        if not sentences_words:
            return tuple()

        tf_metrics = self.summarizer._compute_tf(sentences_words)
        idf_metrics = self.summarizer._compute_idf(sentences_words)

        matrix = self.summarizer._create_matrix(
            sentences_words, self.summarizer.threshold, tf_metrics, idf_metrics
        )
        scores = self.summarizer.power_method(matrix, self.summarizer.epsilon)

        return list(map(str, document.sentences)), list(scores)
