import json
import pprint
from rouge_score import rouge_scorer
from random import choices
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from pathlib import Path
from string import punctuation

from sources.utils import Library

from tqdm import tqdm

from nlp import load_dataset, load_metric


def get_wikinews(
    ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path
):

    if isinstance(ids, int):
        ids = [ids]

    docs = []
    for id in tqdm(ids, desc="Read wikinews articles"):

        with open(wikinews_json_path + "/{:06d}.json".format(id), "r") as json_file:
            try:
                docs.append(json.load(json_file))
            except:
                print(f"{id} doesn't exist")
                docs.append({})

    library = Library(sources_index_path, sources_json_path, sources_html_path)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    for i in tqdm(range(len(docs)), desc="Read sources"):
        sources_text = ""
        for j in range(len(docs[i]["sources"])):
            try:
                docs[i]["sources"][j] = library.get_json(docs[i]["sources"][j])
            except:
                print("source: {} doesn't exist".format(docs[i]["sources"][j]))
                docs[i]["sources"][j] = {}
                continue
            if not "maintext" in docs[i]["sources"][j].keys():
                continue
            if not isinstance(docs[i]["sources"][j]["maintext"], str):
                docs[i]["sources"][j]["score"] = scorer.score("", "")
                continue
            sources_text += " " + docs[i]["sources"][j]["maintext"]
            score = scorer.score(
                " ".join(docs[i]["text"]), docs[i]["sources"][j]["maintext"]
            )
            docs[i]["sources"][j]["score"] = score
        score = scorer.score(" ".join(docs[i]["text"]), sources_text)
        docs[i]["score"] = score

    return docs


def create_dataset(
    recall_threashold,
    hard,
    dataset_path,
    wikinews_index_path,
    wikinews_json_path,
    sources_index_path,
    sources_json_path,
    sources_html_path,
):
    ids = get_ids(wikinews_index_path)
    docs = get_wikinews(
        ids,
        wikinews_json_path,
        sources_index_path,
        sources_json_path,
        sources_html_path,
    )
    num_sources = []
    dataset = []

    remove_by_num_sources = 0
    remove_by_rouge_threahold = 0
    remove_by_clean_conditions = 0

    for doc in tqdm(docs, desc="Write dataset"):
        title = doc["title"]
        summary = "\t".join(doc["text"])
        sources = [
            source["maintext"]
            for source in doc["sources"]
            if isinstance(source, dict)
            and "maintext" in source.keys()
            and source["maintext"] != None
        ]
        if len(sources) == 0:
            remove_by_num_sources += 1
            continue
        if doc["score"]["rouge1"].recall < recall_threashold:
            remove_by_rouge_threahold += 1
            continue
        if not is_clean(summary, "|||".join(sources), len(doc["sources"]), hard=hard):
            remove_by_clean_conditions += 1
            continue
        num_sources.append(len(sources))
        entry = {"title": title, "summary": summary, "sources": "|||".join(sources)}
        dataset.append(entry)

    dataset_train, dataset_test = train_test_split(
        dataset, test_size=0.05, random_state=1
    )

    dataset_train, dataset_val = train_test_split(
        dataset_train, test_size=0.05 / 0.95, random_state=1
    )

    write_dataset(dataset_path, "train", dataset_train)
    write_dataset(dataset_path, "test", dataset_test)
    write_dataset(dataset_path, "validation", dataset_val)

    print(
        "Number of articles removed by:\n - no sources: {}\n - rouge score condition: {}\n - cleaning condition: {}".format(
            remove_by_num_sources,
            remove_by_rouge_threahold,
            remove_by_clean_conditions,
        )
    )

    print(
        "\n\nnumber of articles with:\n - 1 source: {}\n - 2 sources: {}\n - 3 sources: {}\n - 4 sources: {}\n - more sources: {}".format(
            num_sources.count(1),
            num_sources.count(2),
            num_sources.count(3),
            num_sources.count(4),
            len(num_sources)
            - num_sources.count(1)
            - num_sources.count(2)
            - num_sources.count(3)
            - num_sources.count(4),
        )
    )


"""
take example of input and return True is the example is correct or False if it needs to be removed
"""


def is_clean(summary, sources, original_num_sources, hard):

    summary_NoW = len(word_tokenize(summary))
    document_NoW = len(word_tokenize(sources))
    num_sources = len(sources.split("|||"))

    # if summary is less than 1.5 times smaller than document
    if 1.5 * summary_NoW > document_NoW:
        return False

    if not hard:
        return True

    # if summary is less than 2 times smaller than document
    if 2 * summary_NoW > document_NoW:
        return False

    # if summary or document larger than 4000 words
    if summary_NoW > 4000 or document_NoW > 4000:
        return False

    # if more than half of sources was dead
    if num_sources < 0.5 * original_num_sources:
        return False

    return True


def write_dataset(dataset_path, type, dataset):
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    with open(f"{dataset_path}/{type}.jsonl", "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")


def text_to_passages(text):
    def clean(string):
        string = string.replace("\n", "")
        string = string.replace("\t", "")
        return string

    passages = text.split("\n")
    passages = list(map(clean, passages))
    return [passage for passage in passages if passage != ""]


def get_ids(wikinews_index_path, num=-1):
    ids = []
    with open(wikinews_index_path, "r") as f:
        for line in f:
            elems = line.split("\t")
            if len(elems) != 2:
                continue
            ids.append(int(elems[1][:-1]))

    if num == -1:
        return ids

    return choices(ids, k=num)


def stats(dataset_script_path, dataset_cache_path, do_rouge):
    def words_counter(text):
        text = text.translate(str.maketrans(punctuation, " " * len(punctuation)))
        return len(text.split(" "))

    def sentences_counter(text):
        return len(sent_tokenize(text))

    rouge_metric = load_metric("rouge")
    num_sources = []
    sum_num_words = []
    sum_num_sentences = []
    doc_num_words = []
    doc_num_sentences = []

    def compute_stats(example):

        # Rouge score
        prediction = example["clean_document"]
        reference = example["clean_summary"]
        rouge_metric.add(prediction, reference)

        # Number of sources
        num_sources.append(example["document"].count("|||") + 1)

        # Summary length
        sum_num_words.append(words_counter(example["clean_summary"]))
        sum_num_sentences.append(sentences_counter(example["clean_summary"]))

        # Document length
        doc_num_words.append(words_counter(example["clean_document"]))
        doc_num_sentences.append(sentences_counter(example["clean_document"]))

    dataset = load_dataset(
        dataset_script_path, cache_dir=dataset_cache_path, split="train+test+validation"
    )

    dataset = dataset.map(compute_stats)

    if do_rouge:
        rouge_stats = rouge_metric.compute(rouge_types=["rouge1", "rouge2", "rougeL"])

    # Print number of examples
    print(f"The dataset contains {len(dataset)} examples.")

    # Print number of sources stats
    print(
        "number of article with:\n - 1 source: {}\n - 2 sources: {}\n - 3 sources: {}\n - 4 sources: {}\n - more sources: {}".format(
            num_sources.count(1),
            num_sources.count(2),
            num_sources.count(3),
            num_sources.count(4),
            len(num_sources)
            - num_sources.count(1)
            - num_sources.count(2)
            - num_sources.count(3)
            - num_sources.count(4),
        )
    )

    # Print length stats
    print(
        "number of words in document:\t{}\nnumber of sentences in document:\t{}\nnumber of words in summary:\t{}\nnumber of sentences in summary:\t{}\n".format(
            np.mean(doc_num_words),
            np.mean(doc_num_sentences),
            np.mean(sum_num_words),
            np.mean(sum_num_sentences),
        )
    )

    # Print ROUGE stats
    if do_rouge:
        print(
            "Rouge-1 R:\t{}\nRouge-2 R:\t{}\nRouge-L R:\t{}\n".format(
                rouge_stats["rouge1"].mid.recall,
                rouge_stats["rouge2"].mid.recall,
                rouge_stats["rougeL"].mid.recall,
            )
        )
    return None
