"""## Full Evaluation of the Fine-Tuned Model

Runs the fine-tuned model over the Spider test set and scores its generated SQL
two ways:

- Exact match: the generated SQL equals the gold SQL after normalizing
  whitespace and case.
- Execution accuracy: the generated SQL and the gold SQL are run against the
  example's sqlite database and their result sets are compared. This catches
  queries that are syntactically different but semantically equivalent, and
  is the more meaningful of the two metrics.

Usage:
    python finetune_qlora_evaluate.py
    python finetune_qlora_evaluate.py --limit 50
    python finetune_qlora_evaluate.py --output results.json
    python finetune_qlora_evaluate.py --model out/<run dir>/merged
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor, GenerationConfig, pipeline

from spider_prompts import build_prompt_messages, load_tables

DATA_DIR = Path(__file__).parent / "spider_data"
TEST_DB_DIR = DATA_DIR / "test_database"
TEST_TABLES_JSON = DATA_DIR / "test_tables.json"
# Default model: symlink to the merged model of the latest full training run.
DEFAULT_MODEL = Path(__file__).parent / "out" / "text2sql_qlora"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider-format JSON file into a list of example records."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_test_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the test split."""
    return load_json(data_dir / "test.json")


def build_prompt(sample: dict[str, Any], tokenizer, tables: dict[str, dict[str, Any]]) -> str:
    """Render a test sample into the chat-templated prompt the model was trained on
    (see spider_prompts.py), ending with the generation prompt for the model turn."""
    messages = build_prompt_messages(sample["question"], sample["db_id"], tables)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_sql(generated_text: str, prompt: str) -> str:
    """Strip the echoed prompt off a pipeline's generated text, keeping only what
    precedes the turn marker. Batched generation right-pads shorter sequences
    with <pad> tokens after the turn marker, so anything after it is discarded
    rather than just stripped as a suffix."""
    generated = generated_text[len(prompt):]
    return generated.split("<turn|>")[0].strip()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


def run_query(db_id: str, sql: str, db_dir: Path = TEST_DB_DIR) -> tuple[bool, set]:
    """Execute a SQL query and return (succeeded, result rows as a set of tuples)."""
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return True, set(rows)
    except sqlite3.Error:
        return False, set()


def load_pipeline(model_path: Path):
    """Load the merged model, processor, and generation config into a text-generation pipeline."""
    model_id = str(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
    processor = AutoProcessor.from_pretrained(model_id)

    config = GenerationConfig.from_pretrained(model_id)
    config.max_new_tokens = 1024
    config.eos_token_id = [processor.tokenizer.convert_tokens_to_ids("<turn|>")]

    pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer)
    return pipe, processor.tokenizer, config


def evaluate(model_path: Path = DEFAULT_MODEL, limit: int | None = None, batch_size: int = 8) -> list[dict[str, Any]]:
    """Generate and score a SQL prediction for each test example.

    Prompts are fed to the pipeline as a single generator (with `batch_size`)
    instead of one call per example, so generation is actually batched on the
    GPU rather than triggering transformers' "using the pipeline sequentially
    on GPU" warning.
    """
    samples = load_test_dataset()
    if limit:
        samples = samples[:limit]
    tables = load_tables(TEST_TABLES_JSON)

    print(f"Evaluating model: {model_path.resolve()}")
    pipe, tokenizer, config = load_pipeline(model_path)

    prompts = [build_prompt(sample, tokenizer, tables) for sample in samples]
    outputs_iter = pipe((p for p in prompts), generation_config=config, batch_size=batch_size)

    results = []
    for sample, prompt, outputs in tqdm(zip(samples, prompts, outputs_iter), total=len(samples), desc="Evaluating"):
        predicted_sql = extract_sql(outputs[0]["generated_text"], prompt)

        exact_match = normalize_sql(predicted_sql) == normalize_sql(sample["query"])

        pred_ok, pred_rows = run_query(sample["db_id"], predicted_sql)
        gold_ok, gold_rows = run_query(sample["db_id"], sample["query"])
        execution_match = pred_ok and gold_ok and pred_rows == gold_rows

        results.append({
            "db_id": sample["db_id"],
            "question": sample["question"],
            "gold_sql": sample["query"],
            "predicted_sql": predicted_sql,
            "exact_match": exact_match,
            "executable": pred_ok,
            "execution_match": execution_match,
        })

    return results


def display_results(results: list[dict[str, Any]]) -> None:
    """Print a colored per-example breakdown followed by a summary of the metrics."""
    total = len(results)

    for r in results:
        if r["execution_match"]:
            color, status = GREEN, "MATCH"
        elif r["executable"]:
            color, status = YELLOW, "EXEC OK"
        else:
            color, status = RED, "SQL ERROR"

        print(f"{color}[{status}]{RESET} {r['db_id']}: {r['question']}")
        print(f"  gold:      {r['gold_sql']}")
        print(f"  predicted: {r['predicted_sql']}")

    exact_matches = sum(r["exact_match"] for r in results)
    executable = sum(r["executable"] for r in results)
    execution_matches = sum(r["execution_match"] for r in results)

    print()
    print("=" * 60)
    print(f"Total examples:      {total}")
    print(f"Exact match:         {exact_matches}/{total} ({exact_matches / total:.1%})")
    print(f"Executable:          {executable}/{total} ({executable / total:.1%})")
    print(f"Execution accuracy:  {execution_matches}/{total} ({execution_matches / total:.1%})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned text-to-SQL model on the Spider test set."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Merged model directory to evaluate (default: out/text2sql_qlora, the latest full run).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N test examples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save detailed results as JSON.")
    args = parser.parse_args()

    results = evaluate(model_path=args.model, limit=args.limit, batch_size=args.batch_size)
    display_results(results)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
