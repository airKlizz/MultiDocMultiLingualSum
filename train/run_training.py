from scripts.t5 import T5SummarizationTrainer
from scripts.t5_with_title import T5WithTitleSummarizationTrainer
from scripts.bart import BartSummarizationTrainer
from scripts.bert2bert import Bert2BertSummarizationTrainer
from scripts.distilbart import DistilbartSummarizationTrainer


try:
    import wandb
except:
    pass

import argparse

def use_wandb():
    return "wandb" in sys.modules

parser = argparse.ArgumentParser()

parser.add_argument("--bert2bert", action="store_true")
parser.add_argument("--distilbart", action="store_true")
parser.add_argument("--bart", action="store_true")
parser.add_argument("--bart_cnn", action="store_true")
parser.add_argument("--smallt5", action="store_true")
parser.add_argument("--t5", action="store_true")
parser.add_argument("--t5_with_title", action="store_true")

parser.add_argument("--args_file", help="Path to the args json file", type=str)

args = parser.parse_args()


if use_wandb():
    wandb.login()


if args.bert2bert:
    Bert2BertSummarizationTrainer.train(args.args_file)
if args.distilbart:
    DistilbartSummarizationTrainer.train(args.args_file)
if args.bart_cnn:
    BartSummarizationTrainer.train(args.args_file)
if args.bart:
    BartSummarizationTrainer.train(args.args_file)
if args.smallt5:
    T5SummarizationTrainer.train(args.args_file)
if args.t5:
    T5SummarizationTrainer.train(args.args_file)
if args.t5_with_title:
    T5WithTitleSummarizationTrainer.train(args.args_file)
