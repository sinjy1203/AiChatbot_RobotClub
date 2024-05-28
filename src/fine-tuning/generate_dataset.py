import argparse
import random
import re
import tqdm
import json
from functools import partial
from multiprocessing import Pool
from rouge_score import rouge_scorer
from openai import OpenAI

from utils import prompt, generate_context, generate_questions


def get_args():
    parser = argparse.ArgumentParser(description="Quantize AWQ model")

    parser.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--generate_contexts_num", default=100)
    parser.add_argument("--n_shot_contexts", default=5)
    parser.add_argument("--prompt_total_contexts_num", default=15)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_cpus", default=4)

    args = parser.parse_args()
    return args


def main_generate_contexts(client, args):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    progress_bar = tqdm.tqdm(
        total=args.generate_contexts_num, desc="Generating contexts"
    )

    with open("seed_contexts.txt", "r") as f:
        seed_contexts = list(map(lambda x: x.strip(), f.readlines()))
    contexts_pool = seed_contexts[:]
    contexts_pool_tokens = [
        scorer._tokenizer.tokenize(context) for context in contexts_pool
    ]
    progress_bar.update(len(seed_contexts))

    while len(contexts_pool) < args.generate_contexts_num:
        batch_prompt_context_generate = generate_context.batch_prompt(
            seed_contexts, prompt.context_generate, args
        )
        completion = client.completions.create(
            model="models/EEVE-Korean-Instruct-10.8B-v1.0-quantized",
            prompt=batch_prompt_context_generate,
            temperature=1.0,
            top_p=1.0,
            max_tokens=3072,
            stop=[
                f"\n{args.prompt_total_contexts_num}",
                f"{args.prompt_total_contexts_num}.",
                f"{args.prompt_total_contexts_num}.",
            ],
        )
        gen_contexts = generate_context.process_completion(completion)
        filtered_contexts, filtered_contexts_tokens = (
            generate_context.filtering_contexts(
                gen_contexts,
                contexts_pool_tokens,
                scorer,
                args,
            )
        )

        contexts_pool += filtered_contexts
        contexts_pool_tokens += filtered_contexts_tokens
        progress_bar.update(len(filtered_contexts))

    with open("generated_contexts.txt", "w") as f:
        f.write("\n".join(contexts_pool))


def main(client, args):
    with open("generated_contexts.txt", "r") as f:
        contexts = list(map(lambda x: x.strip(), f.readlines()))
    random.shuffle(contexts)

    progress_bar = tqdm.tqdm(total=len(contexts), desc="Generating questions")

    dataset = []
    for idx in range(0, len(contexts), args.batch_size):
        batch_contexts = contexts[idx : idx + args.batch_size]
        batch_prompt_qq_generate = generate_questions.batch_prompt(
            batch_contexts, prompt.prompt_qq_generate
        )

        completion = client.completions.create(
            model="models/EEVE-Korean-Instruct-10.8B-v1.0-quantized",
            prompt=batch_prompt_qq_generate,
            temperature=1.0,
            top_p=1.0,
            max_tokens=3072,
            stop=["\n#", "##", "###"],
        )
        batch_data = generate_questions.process_completion(batch_contexts, completion)

        dataset += batch_data
        progress_bar.update(len(batch_data))

    with open("dataset.jsonl", "w") as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    client = OpenAI(
        base_url=args.llm_base_url,
        api_key="EMPTY",
    )
    main_generate_contexts(client, args)
    main(client, args)
