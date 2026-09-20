"""Prompt construction for the Spider text-to-SQL fine-tuning data.

The model is trained on the prompts that spider_sft_data_prep.ipynb writes into
spider_data_jsonl/*.jsonl, so inference must reproduce them exactly. This module
holds that logic for finetune_qlora_test.py and finetune_qlora_evaluate.py. If the
prompt is changed in the notebook, change it here too; the two are checked against
each other by comparing the output with the generated JSONL files.
"""
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a text-to-SQL system.\n"
    "Use ONLY tables/columns from the schema.\n"
    "Return exactly ONE SQLite SQL query and nothing else.\n"
    "Do NOT include explanations, comments, code fences, or the database id."
)


def load_tables(path: Path) -> dict[str, dict[str, Any]]:
    """Load a Spider tables.json / test_tables.json into {db_id: table info}."""
    with path.open(encoding="utf-8") as f:
        return {t["db_id"]: t for t in json.load(f)}


def schema_to_text(table_info: dict[str, Any]) -> str:
    """Render one database as `table(col, col) ; table(col, ...)`."""
    table_names = table_info["table_names_original"]
    cols_by_table: dict[int, list[str]] = {i: [] for i in range(len(table_names))}
    for tbl_idx, col in table_info["column_names_original"]:
        if tbl_idx == -1:  # the leading "*" pseudo-column
            continue
        cols_by_table[tbl_idx].append(col)
    return " ; ".join(f"{tbl}({', '.join(cols_by_table[i])})" for i, tbl in enumerate(table_names))


def build_user_prompt(question: str, db_id: str, tables: dict[str, dict[str, Any]]) -> str:
    return (
        f"Database id: {db_id}\n"
        f"Schema: {schema_to_text(tables[db_id])}\n"
        f"Question: {question}\n"
        "SQL:"
    )


def build_prompt_messages(question: str, db_id: str, tables: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """The system + user turns the model is prompted with (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, db_id, tables)},
    ]
