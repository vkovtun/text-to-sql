"""## Test Model Inference and generate SQL queries

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

Note: Evaluating generative AI models is not a trivial task since one input can have multiple correct outputs. This guide only focuses on manual evaluation and vibe checks.

Usage:
    python finetune_qlora_test.py                          # random example, test split, latest model
    python finetune_qlora_test.py --split dev
    python finetune_qlora_test.py --index 12                # a specific example instead of a random one
    python finetune_qlora_test.py --seed 42                 # reproducible random pick
    python finetune_qlora_test.py --model out/<run dir>/merged
"""
import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from spider_prompts import (CHAT_TEMPLATE_KWARGS, END_OF_TURN, build_prompt_messages, configure_tokenizer,
                            load_tables, schema_to_text)

# --------------------
# Paths / config
# --------------------
DATA_DIR = Path(__file__).parent / "spider_data"
# Per split: (examples, tables). The dev split shares tables.json with train.
SPLIT_FILES = {
    "dev": (DATA_DIR / "dev.json", DATA_DIR / "tables.json"),
    "test": (DATA_DIR / "test.json", DATA_DIR / "test_tables.json"),
}
DEFAULT_SPLIT = "test"

# Symlink to the merged model of the latest full training run (see finetune_qlora.py)
DEFAULT_MODEL = Path(__file__).parent / "out" / "text2sql_qlora_llama"

MAX_NEW_TOKENS = 256


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider-format JSON file into a list of example records."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_pipeline(model_path: Path, max_new_tokens: int):
    """Load the merged model, tokenizer, and generation config into a text-generation pipeline."""
    model_id = str(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
    # Left padding, matching how the pipeline is called here (single example, no batching)
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_id), padding_side="left")

    config = GenerationConfig.from_pretrained(model_id)
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.eos_token_id = [tokenizer.convert_tokens_to_ids(END_OF_TURN)]
    config.pad_token_id = tokenizer.pad_token_id
    # Safety net for an undertrained checkpoint that fails to emit <|eot_id|>: without
    # this, greedy decoding (do_sample=False) can fall into a repeated-n-gram loop that
    # burns the rest of max_new_tokens instead of trailing off after the real answer.
    config.no_repeat_ngram_size = 3

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fine-tuned model on one Spider example and print its generated SQL."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Merged model directory.")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default=DEFAULT_SPLIT,
                        help="Spider split to sample the example from.")
    parser.add_argument("--index", type=int, default=None,
                        help="Example index to run (default: a random example from the split).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for picking the example (ignored when --index is given).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Generation length cap.")
    args = parser.parse_args()

    data_json, tables_json = SPLIT_FILES[args.split]
    samples = load_json(data_json)
    tables = load_tables(tables_json)

    if args.index is not None:
        test_sample = samples[args.index]
    else:
        test_sample = Random(args.seed).choice(samples)

    print(f"Model: {args.model.resolve()}")
    print(f"Split: {args.split} ({len(samples)} examples)")
    pipe, tokenizer, config = load_pipeline(args.model, args.max_new_tokens)

    # Convert the test example into the prompt the model was trained on, rendered with the Llama chat template
    messages = build_prompt_messages(test_sample["question"], test_sample["db_id"], tables)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
    )
    print(prompt)

    # Generate our SQL query.
    outputs = pipe(text_inputs=prompt, generation_config=config, return_full_text=False)

    print(f"DB ID: ", test_sample["db_id"])
    print(f"Schema:\n", schema_to_text(tables[test_sample["db_id"]]))
    print(f"Question:\n", test_sample["question"])
    print(f"Original Answer:\n{test_sample['query']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'].split(END_OF_TURN)[0].strip()}")


if __name__ == "__main__":
    main()
