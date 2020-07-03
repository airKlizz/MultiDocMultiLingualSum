from scripts.t5 import T5SummarizationTrainer
from scripts.t5_with_title import T5WithTitleSummarizationTrainer
from scripts.bart import BartSummarizationTrainer
from scripts.bert2bert import Bert2BertSummarizationTrainer
from scripts.distilbart import DistilbartSummarizationTrainer


import wandb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--en", action="store_true")
parser.add_argument("--fr", action="store_true")
parser.add_argument("--de", action="store_true")
parser.add_argument("--combine", action="store_true")
parser.add_argument("--bert2bert", action="store_true")
parser.add_argument("--distilbart", action="store_true")
parser.add_argument("--bart", action="store_true")
parser.add_argument("--bart_cnn", action="store_true")
parser.add_argument("--smallt5", action="store_true")
parser.add_argument("--t5", action="store_true")
parser.add_argument("--t5_with_title", action="store_true")

args = parser.parse_args()

lang = None
if args.en:
    lang = "en"
elif args.fr:
    lang = "fr"
elif args.de:
    lang = "de"
elif args.combine:
    lang = "combine"

wandb.login()
if args.bert2bert:
    Bert2BertSummarizationTrainer.train(f"train/args/{lang}_bert2bert.json")
if args.distilbart:
    DistilbartSummarizationTrainer.train(f"train/args/{lang}_distilbart.json")
if args.bart_cnn:
    BartSummarizationTrainer.train(f"train/args/{lang}_bart_cnn.json")
if args.bart:
    BartSummarizationTrainer.train(f"train/args/{lang}_bart.json")
if args.smallt5:
    T5SummarizationTrainer.train(f"train/args/{lang}_smallt5.json")
if args.t5:
    T5SummarizationTrainer.train(f"train/args/{lang}_t5.json")
if args.t5_with_title:
    T5WithTitleSummarizationTrainer.train(f"train/args/{lang}_t5_with_title.json")
