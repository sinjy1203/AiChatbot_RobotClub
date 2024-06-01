import os
import argparse
import numpy as np
import wandb
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer

from utils import prompt

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser(description="Quantize AWQ model")

    parser.add_argument("--model_path", default="beomi/Llama-3-Open-Ko-8B")

    args = parser.parse_args()
    return args


def row2prompt(row: dict):
    prompt_label = prompt.prompt_grade_retrieval + "{label}"
    return prompt_label.format_map(row) + "<|end_of_text|>"


def main(args):
    run = wandb.init(
        project="Grade_Retrieval_LLM", entity="sinjy1203", job_type="train"
    )

    artifact_dataset = run.use_artifact("dataset:latest")
    df = artifact_dataset.get("generated_dataset").get_dataframe()
    df["prompt"] = df.apply(row2prompt, axis=1)

    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    eval_df = df[~msk]

    artifact_dataset_preprocessed = wandb.Artifact(
        name="dataset_preprocessed",
        type="dataset_preprocessed",
        description="preprocessed dataset for LLM fine-tuning.",
    )
    artifact_dataset_preprocessed.add(wandb.Table(data=train_df), "train_dataset")
    artifact_dataset_preprocessed.add(wandb.Table(data=eval_df), "eval_dataset")
    run.log_artifact(artifact_dataset_preprocessed)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=16,
        target_modules=target_modules,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=5.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        learning_rate=1e-5,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to="wandb",
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", quantization_config=nf4_config
    )
    model = get_peft_model(model, lora_config)
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="prompt",
        max_seq_length=256,
        args=training_args,
    )

    trainer.train()
    run.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
