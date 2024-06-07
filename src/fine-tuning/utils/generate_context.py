import random
import re
from functools import partial
from multiprocessing import Pool
from rouge_score import rouge_scorer


def batch_prompt(seed_contexts, prompt_context_generate, args):
    batch_prompt_context_generate = []

    for _ in range(args.batch_size):
        few_shot_contexts = random.sample(seed_contexts, args.n_shot_contexts)
        tmp_prompt = ""
        for idx, few_shot_context in enumerate(few_shot_contexts, start=1):
            tmp_prompt += f"{idx}. {few_shot_context}\n"
        batch_prompt_context_generate += [prompt_context_generate + tmp_prompt]

    return batch_prompt_context_generate


def process_completion(completion):
    gen_contexts = []
    for completion_choice in completion.choices:
        if completion_choice.finish_reason == "length":
            continue
        gen_contexts += re.findall(r"\d+\.\s*(.+)", completion_choice.text)
    return gen_contexts


def filtering_contexts(gen_contexts, contexts_pool_tokens, scorer, args):
    filtered_contexts = []
    filtered_contexts_tokens = []
    for gen_context in gen_contexts:
        gen_context_tokens = scorer._tokenizer.tokenize(gen_context)
        with Pool(args.num_cpus) as p:
            scores = p.map(
                partial(rouge_scorer._score_lcs, gen_context_tokens),
                contexts_pool_tokens,
            )

        scores = [score.fmeasure for score in scores]
        if max(scores) > 0.7:
            continue

        filtered_contexts += [gen_context]
        filtered_contexts_tokens += [gen_context_tokens]
    return filtered_contexts, filtered_contexts_tokens
