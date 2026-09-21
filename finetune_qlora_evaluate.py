"""## Generate Spider predictions with the fine-tuned model

Runs the merged fine-tuned model over a Spider split and writes one predicted SQL
query per line, in dataset order. That is the format the official Spider
evaluation script expects, and that script produces the exact-match and
execution-accuracy scores:

    python finetune_qlora_evaluate.py
    python evaluation.py --gold spider_data/test_gold.sql \\
        --pred out/eval/pred_test_eval_qlora.sql \\
        --db spider_data/test_database --table spider_data/test_tables.json --etype all

Usage:
    python finetune_qlora_evaluate.py                        # full test split
    python finetune_qlora_evaluate.py --split dev
    python finetune_qlora_evaluate.py --limit 20             # quick check
    python finetune_qlora_evaluate.py --model out/<run dir>/merged --output out/eval/other.sql
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor, GenerationConfig, pipeline

from spider_prompts import build_prompt_messages, load_tables

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

# Merged (base + adapter) model to run, and where the predictions go.
DEFAULT_MODEL = Path(__file__).parent / "out" / "text2sql_qlora_euler"
OUTPUT_DIR = Path(__file__).parent / "out" / "eval"

MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
# Flush the predictions file to disk every N examples so a crash loses little work.
FLUSH_EVERY = 100
# Gemma's end-of-turn marker, which ends the generated answer.
END_OF_TURN = "<turn|>"


def default_output_path(split: str) -> Path:
    return OUTPUT_DIR / f"pred_{split}_qlora_eval.sql"


def load_pipeline(model_path: Path):
    """Load the merged model, processor, and generation config into a text-generation pipeline."""
    model_id = str(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
    processor = AutoProcessor.from_pretrained(model_id)

    config = GenerationConfig.from_pretrained(model_id)
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False
    config.eos_token_id = [processor.tokenizer.convert_tokens_to_ids(END_OF_TURN)]

    pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer)
    return pipe, processor.tokenizer, config


def build_prompt(sample: dict[str, Any], tokenizer, tables: dict[str, dict[str, Any]]) -> str:
    """Render a sample into the chat-templated prompt the model was trained on
    (see spider_prompts.py, which includes the DB schema), ending with the
    generation prompt for the model turn."""
    messages = build_prompt_messages(sample["question"], sample["db_id"], tables)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_sql(generated_text: str, prompt: str) -> str:
    """Turn a pipeline's generated text into a single-line SQL query.

    The pipeline echoes the prompt, and batched generation right-pads shorter
    sequences with <pad> tokens after the end-of-turn marker, so anything after
    the marker is discarded rather than just stripped as a suffix. The query is
    collapsed onto one line (and stripped of a trailing semicolon) because the
    Spider evaluation script reads one query per line."""
    answer = generated_text[len(prompt):].split(END_OF_TURN)[0]
    return " ".join(answer.split()).rstrip(";").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the fine-tuned model's SQL predictions for a Spider split, one query per line."
    )
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default=DEFAULT_SPLIT, help="Spider split to run.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Merged model directory.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Predictions file (default: out/eval/pred_<split>_eval_qlora.sql).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Generation batch size.")
    args = parser.parse_args()

    data_json, tables_json = SPLIT_FILES[args.split]
    samples = json.loads(data_json.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[:args.limit]
    tables = load_tables(tables_json)

    output_path = args.output or default_output_path(args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {args.model.resolve()}")
    print(f"Split:  {args.split} ({len(samples)} examples)")
    print(f"Output: {output_path}")
    pipe, tokenizer, config = load_pipeline(args.model)

    # Prompts are fed to the pipeline as one generator (with batch_size) so
    # generation is batched on the GPU instead of running example by example.
    prompts = [build_prompt(sample, tokenizer, tables) for sample in samples]
    outputs_iter = pipe((p for p in prompts), generation_config=config, batch_size=args.batch_size)

    with output_path.open("w", encoding="utf-8") as f:
        for count, (prompt, outputs) in enumerate(
            tqdm(zip(prompts, outputs_iter), total=len(prompts), desc="Generating"), start=1
        ):
            # Always write a line, even an empty one, to keep predictions aligned with the gold file.
            f.write(extract_sql(outputs[0]["generated_text"], prompt) + "\n")
            if count % FLUSH_EVERY == 0:
                f.flush()
                os.fsync(f.fileno())

    print(f"Wrote {len(prompts)} predictions to {output_path}")


if __name__ == "__main__":
    main()
