# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""WikinewsSum-EN dataset."""

from __future__ import absolute_import, division, print_function

import os

import json

import nlp


_CITATION = """
Anonymous submission
"""

_DESCRIPTION = """
English Wikinews dataset
"""

_PATH = (
    "dataset/en-wiki-multi-news/"
)

URL = (
    "https://drive.google.com/uc?export=download&id=1VvhCPBCdeyP1Z-kBwVjfhU-K3HD40VHg"
)

_TITLE = "title"
_DOCUMENT = "document"
_SUMMARY = "summary"
_CLEAN_DOCUMENT = "clean_document"
_CLEAN_SUMMARY = "clean_summary"


class WikinewsSumEN(nlp.GeneratorBasedBuilder):
    """WikinewsSum-EN dataset."""

    VERSION = nlp.Version("1.0.0")

    def _info(self):
        info = nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    _TITLE: nlp.Value("string"),
                    _DOCUMENT: nlp.Value("string"),
                    _SUMMARY: nlp.Value("string"),
                    _CLEAN_DOCUMENT: nlp.Value("string"),
                    _CLEAN_SUMMARY: nlp.Value("string"),
                }
            ),
            # supervised_keys=(_TITLE, _DOCUMENT, _SUMMARY),
            homepage="",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #data_path = _PATH
        data_path = dl_manager.download_and_extract(URL)
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "train.jsonl")},
            ),
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "validation.jsonl")},
            ),
            nlp.SplitGenerator(
                name=nlp.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "test.jsonl")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path) as f:
            for i, line in enumerate(f):
                elem = json.loads(line)
                yield i, {
                    _TITLE: elem["title"],
                    _DOCUMENT: elem["sources"],
                    _SUMMARY: elem["summary"],
                    _CLEAN_DOCUMENT: self.clean_document(elem["sources"]),
                    _CLEAN_SUMMARY: self.clean_summary(elem["summary"]),
                }

    def clean_summary(self, summary):
        summary = summary.replace("\t", " ")
        summary = summary.replace("\n", " ")
        return summary

    def clean_document(self, document):
        document = document.replace("|||", " ")
        document = document.replace("\n", " ")
        return document
