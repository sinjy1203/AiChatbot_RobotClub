import os
from tqdm import tqdm
import re
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import wandb
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils import prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parsing(pred):
    pattern = r"### 평가: \n(예|아니오)"
    match = re.search(pattern, pred)
    return match.group(1)


def get_args():
    parser = argparse.ArgumentParser(description="Quantize AWQ model")

    parser.add_argument("--run_name", default="EEVE-Korean-Instruct-10.8B-v1.0")

    args = parser.parse_args()
    return args


def evaluate(model, tokenizer, df):
    pred_lst = []
    for idx in tqdm(range(len(df))):
        input_ids = tokenizer(
            prompt.prompt_grade_retrieval.format_map(df.iloc[idx].to_dict()),
            return_tensors="pt",
        ).input_ids

        generation_output = model.generate(
            input_ids=input_ids,
            max_length=200,
            eos_token_id=[
                tokenizer.convert_tokens_to_ids("#"),
                tokenizer.convert_tokens_to_ids("##"),
                tokenizer.convert_tokens_to_ids("###"),
                tokenizer.eos_token_id,
            ],
            pad_token_id=tokenizer.eos_token_id,
        )

        pred = tokenizer.decode(generation_output[0])
        res = parsing(pred)
        pred_lst += [1 if res == "예" else 0]

    labels = df["label"].to_numpy()
    labels[labels == "예"] = 1
    labels[labels == "아니오"] = 0
    labels = labels.tolist()

    return {
        "accuracy": accuracy_score(labels, pred_lst),
        "precision": precision_score(labels, pred_lst),
        "recall": recall_score(labels, pred_lst),
        "f1": f1_score(labels, pred_lst),
    }


def main(args):
    run = wandb.init(project="Grade_Retrieval_LLM", entity="sinjy1203")

    artifact_dataset = run.use_artifact("dataset_preprocessed:latest")
    artifact_model = run.use_artifact(f"model-{args.run_name}:latest", type="model")

    train_df = artifact_dataset.get("train_dataset").get_dataframe()
    eval_df = artifact_dataset.get("eval_dataset").get_dataframe()

    model_path = artifact_model.download()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=nf4_config, device_map="auto"
    )

    train_metrics = evaluate(model, tokenizer, train_df)
    eval_metrics = evaluate(model, tokenizer, eval_df)

    for key in train_metrics:
        artifact_model.metadata["train_" + key] = train_metrics[key]
        artifact_model.metadata["eval_" + key] = eval_metrics[key]

    artifact_model.save()


if __name__ == "__main__":
    args = get_args()
    main(args)
