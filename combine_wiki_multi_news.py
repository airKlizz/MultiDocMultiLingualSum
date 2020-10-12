# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace datasets Authors.
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
"""Combine Wiki Multi News dataset."""

from __future__ import absolute_import, division, print_function

import os
import random

import json

import datasets


_CITATION = """
DFKI
"""

_DESCRIPTION = """
Combine Wikinews dataset
"""

_EN_URL = (
    "https://drive.google.com/uc?export=download&id=1VvhCPBCdeyP1Z-kBwVjfhU-K3HD40VHg"
)
_DE_URL = (
    "https://drive.google.com/uc?export=download&id=1DGnbUY9zwiThTdgUvVTSAvSVHoloCgun"
)
_FR_URL = (
    "https://drive.google.com/uc?export=download&id=1n_JFVzQuCi9srpukYyQojWjyb06d-eFF"
)

_VERSION = "version"
_TITLE = "title"
_DOCUMENT = "document"
_SUMMARY = "summary"
_CLEAN_DOCUMENT = "clean_document"
_CLEAN_SUMMARY = "clean_summary"


class CombineWikiMultiNews(datasets.GeneratorBasedBuilder):
    """Multi-News dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _VERSION: datasets.Value("string"),
                    _TITLE: datasets.Value("string"),
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                    _CLEAN_DOCUMENT: datasets.Value("string"),
                    _CLEAN_SUMMARY: datasets.Value("string"),
                }
            ),
            # supervised_keys=(_TITLE, _DOCUMENT, _SUMMARY),
            homepage="https://github.com/airKlizz/MultiDocMultiLingualSum",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        en_data_path = dl_manager.download_and_extract(_EN_URL)
        de_data_path = dl_manager.download_and_extract(_DE_URL)
        fr_data_path = dl_manager.download_and_extract(_FR_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "en_path": os.path.join(en_data_path, "train.jsonl"),
                    "de_path": os.path.join(de_data_path, "train.jsonl"),
                    "fr_path": os.path.join(fr_data_path, "train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "en_path": os.path.join(en_data_path, "validation.jsonl"),
                    "de_path": os.path.join(de_data_path, "validation.jsonl"),
                    "fr_path": os.path.join(fr_data_path, "validation.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "en_path": os.path.join(en_data_path, "test.jsonl"),
                    "de_path": os.path.join(de_data_path, "test.jsonl"),
                    "fr_path": os.path.join(fr_data_path, "test.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, en_path=None, de_path=None, fr_path=None, seed=42):
        """Yields examples."""
        random.seed(seed)
        examples = []
        for version, path in zip(["en", "de", "fr"], [en_path, de_path, fr_path]):
            with open(path) as f:
                for i, line in enumerate(f):
                    elem = json.loads(line)
                    examples.append(
                        {
                            _VERSION: version,
                            _TITLE: elem["title"],
                            _DOCUMENT: elem["sources"],
                            _SUMMARY: elem["summary"],
                            _CLEAN_DOCUMENT: self.clean_document(elem["sources"]),
                            _CLEAN_SUMMARY: self.clean_summary(elem["summary"]),
                        }
                    )
        random.shuffle(examples)
        for i, example in enumerate(examples):
            yield i, example

    def clean_summary(self, summary):
        summary = summary.replace("\t", " ")
        summary = summary.replace("\n", " ")
        return summary

    def clean_document(self, document):
        document = document.replace("|||", " ")
        document = document.replace("\n", " ")
        return document
