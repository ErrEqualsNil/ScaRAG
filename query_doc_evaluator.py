import json
import os

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils.evaluate import is_resp_match_answer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    train_dataset, eval_dataset = load_data(
        "data/query_content_train_dataset/dataset.jsonl",
        "data/total/popqa_qd_pairs_with_resp.jsonl"
    )
    # 加载模型和分词器
    model_name = "google-t5/t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trust_remote_code=True, device_map="cuda:0")

    train_dataset, eval_dataset = (
        train_dataset.map(preprocess_data, fn_kwargs={"tokenizer": tokenizer}, batched=True),
        eval_dataset.map(preprocess_data, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    )

    # 设置训练参数
    training_args = TrainingArguments(
        run_name="query_content_matching_llm",
        output_dir="data4/model_outputs/query_content_evaluator",
        logging_steps=10,

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        eval_strategy="steps",
        eval_steps=2000,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        num_train_epochs=6,

        optim="adafactor",
        learning_rate=1e-4,
        warmup_steps=0,
        lr_scheduler_type="linear",

        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 开始训练
    trainer.train()



if __name__ == '__main__':
    train()
