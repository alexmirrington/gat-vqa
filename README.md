# Graph Attention Networks for Compositional Visual Question Answering

<p align="center">
  <a href="https://github.com/alexmirrington/graphgen/actions?query=branch%3Adevelop+workflow%3Atests">
    <img src="https://github.com/alexmirrington/graphgen/workflows/tests/badge.svg" alt="Tests status" />
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
  </a>
</p>

----------------------

## Getting started

### Prerequisites

Python: >=3.8
CUDA: >=10.2

Previous Python and CUDA versions may be compatible, but have not been tested.

### Install requirements

It is strongly recommended to install all requirements in a virtual environment,
as some required pytorch extensions are sensitive to both `torch` and system CUDA versions.

Make a clean virtual environment:

```Bash
python -m venv .virtualenvs/graphgen
source .virtualenvs/graphgen/bin/activate
```

Check your current system CUDA install version with `nvcc`:

```Bash
nvcc --version
```

If your system cuda version is the same as the latest CUDA version supported by `torch`,
you can simply run `make install` to install all dependencies.

If not, you should install `torch` and `torchvision` binaries that were built on the same CUDA version as your system CUDA version, either [prebuilt](https://pytorch.org/get-started/locally/) or from source. You can then run `make install` to install remaining dependencies.

For CUDA 11.0 support, you need to build `torch`, `torchvision`, `torchtext` and all [torch-geometric dependencies](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) from source until CUDA 11.0 is officially supported. Remaining requirements can be installed using the command `make requirements`.

### Set up Weights and Biases

Weights and Biases (`wandb`) is a superb tool for tracking machine learning projects, providing a myriad of time-saving features such as live metric logging, dataset/model artifact versioning and distributed hyperparameter tuning. Whilst the Python package is required for this code to run, you do not need to make an account if you prefer not to sign up. The first time you run `main.py`, simply choose whether to log in to an existing account, sign up for a `wandb` account or disable experiment tracking.

If you would like to track experiments and use dataset and model versioning capabilities, you can also log in anonymously like so:

```Bash
wandb login --anonymously
```

## Running the code

There are three types of jobs that `main.py` supports: `preprocess`, `train` and `predict`. To run the code, simply provide a job type, a config file and optionally a `--sync` flag to tell `wandb` to sync your run to the cloud, _e.g._

```Bash
python main.py --config config/gqa/mac/vqa_qn_lstm_sg_gat.json --job train --sync
```

Each of the three jobs are explained in detail in the folloring subsections

### Preprocessing

You can either download the GQA dataset yourself and run a preprocessing job to generate dependency parser outputs and other required preprocessed data, or use an existing preprocessed dataset (stored as a `wandb` artifact) specified by the `config.data.artifact` string. To preprocess the dataset yourself, run the following:

```Bash
python main.py --config config/gqa/mac/vqa_qn_lstm_sg_gat.json --job preprocess --sync
```

The preprocessed dataset will be stored in the `cache` directory and uploaded to `wandb` as an artifact.

### Training

Choose a model to train from the `config` folder and pass it to the `--config` parameter like so:

```Bash
python main.py --config config/gqa/mac/vqa_qn_lstm_sg_gat.json --job train --sync
```

### Model Predictions and Evaluation

First, gather model predictions. Note that if you want to evaluate GQA consistency, we need predictions on the `all` version of the GQA dataset, not just the `balanced` dataset. This is the default behaviour for all model evaluation, however evaluation on all samples takes a while. Similarly to training a model, we specify the model config but set the job to `predict` like so:

```Bash
python main.py --config config/gqa/mac/vqa_qn_lstm_sg_gat.json --job predict --sync
```

This will do two things:

- Load the train, val and test datasets from according to the `training.data` field in the config and dump a list of all question IDs from those datasets to a file for use when computing metrics later. This is essential when using a the first half of the GQA validation set for validation and the second half for testing, since `eval.py` requires a list of IDs to include or exclude in those cases. If evaluating on the full validation or training set, we do not need these IDs.
- Gather model predictions for the train, val and test datasets as specified by the `config.prediction.data` fields. By default, this will evaluate the model on the unbalanced (all) GQA train and val splits.

After gathering model predictions, we can evaluate the model on various GQA metrics, with `eval.py`; by default, we train models on GQA balanced train, eval on the first half of GQA balanced val, and test on the second half of GQA balanced val. If using this setup, we get the train, val and test metrix with the following commands:

```Bash
python eval.py --tier train --predictions train_predictions.json --consistency
python eval.py --tier val --predictions val_predictions.json --include-ids val_ids.json --exclude-ids test_ids.json --consistency
python eval.py --tier val --predictions test_predictions.json --include-ids test_ids.json --exclude-ids val_ids.json --consistency
```

## Running Unit Tests

To run unit tests, simply run the `test` target from the `Makefile` using `make`:

```Bash
make test
```

This target wraps `pytest`, and logs code coverage results.
