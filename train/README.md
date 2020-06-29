# Training of summarization models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15mhkhI7sNATanwGHqmh9ZO69AngAUg37?usp=sharing)

Models are trained on WikinewsSum by default but this can be changed by changing the training parameters.

All the training scripts are in the folder ``scripts/`` and all the training parameters are in the folder ``args/``.

To use training parameters by default, just run:

```bash
cd path/to/MultiDocMultiLingualSum/
python train/run_training --model --version
```
where ``model`` is the model you want to train. Models available are:

- ``bert2bert``,
- ``bart``,
- ``bart_cnn``,
- ``t5``,
- ``t5_with_title``.

and ``version`` is the version of the dataset. Versions available are:

- ``en``,
- ``de``,
- ``fr``,
- ``combine``,

So to train ``bart`` on the English version of WikinewsSum, you have to run: ``python train/run_training --bart --en``.

> Remark: All combinations are not working. Example: ``bart_cnn`` with ``de`` as Bart fine-tuned on CNN-DM has been made for English.

To add you model, create a ``json`` file in ``args/`` with your training parameters.
