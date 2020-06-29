# Generating Extended and Multilingual Summaries with Pre-trained Transformers

Code to reproduce data, models and results of the paper **Generating Extended and Multilingual Summaries with Pre-trained Transformers**.

## WikinewsSum

### Reproduce the dataset

All the code to create WikinewsSum and reproduce stats and explaination are in the [``dataset``](/dataset) folder.

### Load the dataset

Raw data of each version of the dataset are available [here](https://drive.google.com/drive/folders/1805OtY_T0lVL3xCSGPvBdQpAkvIl6eO4?usp=sharing).

You can also load the dataset with the [HuggingFace nlp library](https://github.com/huggingface/nlp) using ``en_wiki_multi_news.py`` for the English version, ``de_wiki_multi_news.py`` for the German version or ``fr_wiki_multi_news.py`` for the French one.

For load the WikinewsSum-EN, run:

```python
from nlp import load_dataset

dataset = load_dataset('en_wiki_multi_news.py', cache_dir='dataset/.en-wiki-multi-news-cache')

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']
```

## Models

Training models are available as HugginFace models [here](https://huggingface.co/models?search=airKlizz).

Implementation code and training scripts are in the [``train``](/train) folder.

For example, you can use BART fine-tuned on WikinewsSum-EN as follow:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# Load model
model = AutoModelWithLMHead.from_pretrained("airKlizz/bart-large-multi-en-wiki-news")

# Prepare inputs
inputs = tokenizer.encode_plus(TEXT_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

# Summarize
outputs = model.generate(
  input_ids=inputs['input_ids'], 
  attention_mask=inputs['attention_mask'], 
  max_length=400, 
  min_length=150, 
  length_penalty=2.0, 
  num_beams=4, 
  early_stopping=True
)

# Decode
summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(summary)
```

## Results

All extractive and abstractive models implementations and evaluation scripts are in the [``evaluate``](/evaluate) folder.

We create an summarization evaluation environement easy to use for all models and all datasets. You can find more details in the [``evaluate``](/evaluate) folder.

## Demo

![](demo/demo.gif)

A demo will be available soon.

<!---

## Related Work

### English Multi Document Summarization

* [Multi-News: a Large-Scale Multi-Document SummarizationDataset and Abstractive Hierarchical Model](https://arxiv.org/pdf/1906.01749.pdf) | [github](https://github.com/Alex-Fabbri/Multi-News)\
Multi-news dataset from [newser](https://www.newser.com/) available.
* [Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf) | [github](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum)\
Great only decoder architecture with memory improvements to allow for input tokens but train a model from scratch would need to much data.
* [Hierarchical Transformers for Multi-Document Summarization](https://arxiv.org/pdf/1905.13164.pdf) | [github](https://github.com/nlpyang/hiersumm)\
WikiSum dataset avaible.
* [Towards Automatic Construction of News Overview Articles by NewsSynthesis](https://www.aclweb.org/anthology/D17-1224.pdf)\
Use part of (top 100 articles in length) english Wikinews as test dataset and evaluate several methods but not deep learning methods.
* [A Zipf-Like Distant Supervision Approachfor Multi-document SummarizationUsing Wikinews Articles](https://www.researchgate.net/profile/Felipe_Bravo-Marquez/publication/233158220_A_Zipf-Like_Distant_Supervision_Approach_for_Multi-document_Summarization_Using_Wikinews_Articles/links/0fcfd509bfcdcf274f000000/A-Zipf-Like-Distant-Supervision-Approach-for-Multi-document-Summarization-Using-Wikinews-Articles.pdf)\
Perform extractive summarization on english Wikinews using Zipf-Like Distant Supervision (don't know how it works).
* [Multi-Topic Multi-Document Summarizer](https://arxiv.org/pdf/1401.0640.pdf)\
The paper mentions: "The data set is available in 7 languages including Arabic (http://www.nist.gov/tac/2011/Summarization/). It was derived from publicly available WikiNews English texts.". But I didn't find the dataset.
* [GameWikiSum: a Novel Large Multi-Document Summarization Dataset](https://arxiv.org/pdf/2002.06851.pdf) | [github](https://github.com/Diego999/GameWikiSum)\
Interesting work on video-game multi-document summarization which compare several methods (extractive and abstractive). Similar to what we want to do but for video-game documents.

### Multi-Lingual Summarization

* [Sequential Transfer Learning in NLP forGerman Text Summarization](http://ceur-ws.org/Vol-2458/paper8.pdf)\
Use BERT as encoder.
* [Abstract Text Summarization: A Low Resource Challenge](https://www.aclweb.org/anthology/D19-1616.pdf)\
Data augmentation for summarization. Use OpenNMT-py transformers.
* [MultiLing 2015: Multilingual Summarization of Single andMulti-Documents, On-line Fora, and Call-center Conversations](https://www.aclweb.org/anthology/W15-4638.pdf)\
They speach about a multilingual multi-Document Summarization dataset with 10 documents per topic and 15 topics. Again I didn't find it.
* [AllSummarizer system at MultiLing 2015:Multilingual single and multi-document summarization](https://www.aclweb.org/anthology/W15-4634.pdf) | [github](https://github.com/kariminf/AllSummarizer)\
Summarizer language and domain independent evaluate on  *MultiLing 2015 multi-document testing dataset* which seems great but I can't find it.

### English Abstract Summarization

* [BART: Denoising Sequence-to-Sequence Pre-training for NaturalLanguage Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) | [github](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)\
Great results on CNN-DM.
* [Exploring the Limits of Transfer Learning with aUnified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) | [github](https://github.com/google-research/text-to-text-transfer-transformer)\
Great results on CNN-DM.
* [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) | [github](https://github.com/nlpyang/PreSumm)\
Use BERT as encoder and a non-trained decoder. Two optimizers to avoid overfitting of BERT.
* [Transforming Wikipedia into Augmented Datafor Query-Focused Summarization](https://arxiv.org/pdf/1911.03324.pdf)\
/|Extractive summarization|\] Use Bert for topic based summary.

### Evaluation

* [EASY-M: Evaluation System for Multilingual Summarizers](https://www.aclweb.org/anthology/W19-89.pdf#page=63) | [web](https://summaryevaluation.azurewebsites.net/home)\
Online system that evaluate summaries with many metrics. Can be use to compare with our metrics.

--->
