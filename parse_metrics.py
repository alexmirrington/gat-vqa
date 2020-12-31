"""Script for parsing results from eval.py and syncing to wandb."""
import argparse
import os
import re
import tempfile
from typing import Any, Dict

import wandb


def main(args: argparse.Namespace) -> None:
    """Parse and sync results."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    api = wandb.Api()
    run = api.run(f"alexmirrington/graphgen/{args.id}")
    filenames = {
        "train": run.file("train_metrics.txt"),
        "val": run.file("val_metrics.txt"),
        "test": run.file("test_metrics.txt"),
    }
    metrics = {}
    metric_keyswap = {
        "attr": "attribute",
        "cat": "category",
        "obj": "object",
        "rel": "relation",
    }
    step_flag = False
    words_flag = False
    step_data: Dict[str, Any] = {}
    word_data: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for split, filename in filenames.items():
            step_data[split] = []
            word_data[split] = []
            file_ = filename.download(tempdir)
            lines = file_.readlines()
            for line in lines:
                line = line.rstrip()
                if "python" in line or line == "":
                    continue
                # Match all but steps and wordsnumber
                match = re.match(r"\s*([A-Za-z]+): ([0-9]+\.[0-9]+)%?", line)
                if match is not None:
                    metric_key = match.group(
                        1
                    ).lower()  # pylint: disable=consider-using-get
                    if metric_key in metric_keyswap:
                        metric_key = metric_keyswap[metric_key]
                    metric_val = float(match.group(2))
                    metric_val = metric_val / 100 if metric_val > 1 else metric_val
                    metrics[f"{split}/{metric_key}"] = metric_val
                    continue
                # Match accuracy per reasoning steps
                if line == "Accuracy / steps number:":
                    step_flag = True
                    continue
                if step_flag:
                    match = re.match(
                        r"\s*([0-9]+): ([0-9]+\.[0-9]+)% \(([0-9]+) questions\)", line
                    )
                    if match is None:
                        step_flag = False
                    else:
                        step = int(match.group(1))
                        accuracy = float(match.group(2)) / 100
                        count = int(match.group(3))
                        step_data[split].append([step, accuracy, count])
                        continue
                # Match accuracy per word count steps
                if line == "Accuracy / words number:":
                    words_flag = True
                    continue
                if words_flag:
                    match = re.match(
                        r"\s*([0-9]+): ([0-9]+\.[0-9]+)% \(([0-9]+) questions\)", line
                    )
                    if match is None:
                        words_flag = False
                    else:
                        word = int(match.group(1))
                        accuracy = float(match.group(2)) / 100
                        count = int(match.group(3))
                        word_data[split].append([word, accuracy, count])
                        continue
                # No match, skip
                print(f"skipping: {line}")

    for key, val in metrics.items():
        run.summary[key] = val
    run.summary.update()

    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = args.id
    resumed = wandb.init()
    tables = {}
    for split, table in word_data.items():
        tables[f"{split}/per_word_count_accuracy"] = wandb.Table(
            data=table, columns=["Word Count", "Accuracy", "Question Count"]
        )
    for split, table in step_data.items():
        tables[f"{split}/per_step_count_accuracy"] = wandb.Table(
            data=table, columns=["Reasoning Steps", "Accuracy", "Question Count"]
        )
    resumed.log(tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=str, required=True, help="The run id to pull metrics from."
    )
    main(parser.parse_args())
