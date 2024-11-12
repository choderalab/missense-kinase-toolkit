#!/usr/bin/env python

import os

import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from utils import (  # save_stats,
    compute_metrics,
    load_csv2dataset,
    parsearg_utils,
    save_csv2csv,
)


def main():
    args = parsearg_utils()

    if args.wandbRun != "":
        csv_out = args.wandbRun.split("-")[1] + "_data.csv"
    else:
        csv_out = None

    # load data
    df = pd.read_csv(os.path.join(args.path, args.inputData))
    csv_out = save_csv2csv(
        df=df,
        path=args.path,
        csv_name=csv_out,
        seed=args.seed,
        col_seq=args.columnSeq,
    )

    # split data
    ds_val, ds_train = load_csv2dataset(
        args.path,
        args.kFold,
        csv_name=csv_out,
    )

    # full dataset
    file_path = os.path.join(args.path, "assets", csv_out)

    ds_full = load_dataset("csv", data_files=file_path)
    ds_val.append(ds_full["train"])
    ds_train.append(ds_full["train"])

    # encode data
    max_len = df[args.columnSeq].str.len().max()
    dict_token_args = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_len,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # https://huggingface.co/docs/datasets/v1.5.0/processing.html
    col_seq = "seq"
    encode_val = [
        dataset.map(lambda x: tokenizer(x[col_seq], **dict_token_args), batched=True)
        for dataset in ds_val
    ]
    encode_val = [dataset.remove_columns([col_seq]) for dataset in encode_val]
    encode_train = [
        dataset.map(lambda x: tokenizer(x[col_seq], **dict_token_args), batched=True)
        for dataset in ds_val
    ]
    encode_train = [dataset.remove_columns([col_seq]) for dataset in encode_train]

    # create dictionary for encoding
    dict_names = [f"fold-{int(i) + 1}" for i in range(args.kFold)]
    dict_names.append("full")

    dict_encode = {
        name: {"val": val, "train": train}
        for name, val, train in zip(dict_names, encode_val, encode_train)
    }

    if args.wandbProject != "":
        os.environ["WANDB_PROJECT"] = args.wandbProject
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    for key, dataset in dict_encode.items():
        # create sub-directories for each fold
        if args.wandbRun != "":
            path_wandb = os.path.join(args.path, args.wandbRun)
            if not os.path.exists(path_wandb):
                os.mkdir(path_wandb)
            path_main = os.path.join(path_wandb, str(key))
        else:
            path_main = os.path.join(args.path, str(key))
        path_results = os.path.join(path_main, "results")
        path_logs = os.path.join(path_main, "logs")
        for path in [path_main, path_results, path_logs]:
            if not os.path.exists(path):
                os.mkdir(path)

        # training arguments
        dict_training_args = {
            "learning_rate": args.learningRate,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.tBatch,
            "per_device_eval_batch_size": args.vBatch,
            "warmup_steps": args.warmup,
            "weight_decay": args.weightDecay,
            "output_dir": path_results,
            "overwrite_output_dir": args.overwrite,
            "save_total_limit": args.saveLim,
            "evaluation_strategy": args.evalStrategy,
            "save_strategy": args.saveStrategy,
            "load_best_model_at_end": args.loadBest,
            "logging_dir": path_logs,
            "logging_steps": args.loggingSteps,
        }
        if args.wandbProject != "":
            dict_training_args["report_to"] = "wandb"
        if args.wandbRun != "":
            dict_training_args["run_name"] = args.wandbRun
        training_args = TrainingArguments(**dict_training_args)

        # load model; num_labels=1 for regression
        model = EsmForSequenceClassification.from_pretrained(
            args.model, num_labels=1, problem_type="regression"
        )

        # set trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dataset["train"].with_format("torch"),
            eval_dataset=dataset["val"].with_format("torch"),
        )
        # try:
        #     trainer.train(resume_from_checkpoint=True)
        # except:
        #     trainer.train()
        trainer.train()

        pd.DataFrame(trainer.state.log_history).to_csv(
            os.path.join(path_logs, f"{key}_trainer_state_log.csv"), index=False
        )
        # save_stats(trainer, path)


if __name__ == "__main__":
    main()
