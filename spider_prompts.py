"""Prompt construction for the Spider text-to-SQL fine-tuning data.

The model is trained on the prompts that spider_sft_data_prep.ipynb writes into
spider_data_jsonl/*.jsonl, so inference must reproduce them exactly. This module
holds that logic for finetune_qlora.py, finetune_qlora_test.py and finetune_qlora_evaluate.py. If the
prompt is changed in the notebook, change it here too; the two are checked against
each other by comparing the output with the generated JSONL files.
"""
import json
from pathlib import Path
from typing import Any

## Llama 3 chat-format details shared by training and inference; they must match.                                                                                    
# END_OF_TURN = "<|eot_id|>"  # closes every turn, so it is also the answer's final label                                                                             
# PAD_TOKEN = "<|finetune_right_pad_id|>"  # Llama tokenizers ship without a pad token                                                                                
## The Llama 3.x chat template writes a "Today Date" line into the system header (3.2                                                                                
## defaults it to the current date). A fixed value keeps prompts identical across days.                                                                              
#CHAT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}            

# Chat-format details shared by training and inference; they must match. These are
# set for Qwen (Qwen2.5 / Qwen3 Instruct, ChatML format). To go back to Llama 3, use:
#   END_OF_TURN = "<|eot_id|>"
#   PAD_TOKEN = "<|finetune_right_pad_id|>"  # Llama tokenizers ship without a pad token
#   CHAT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}  # fixed "Today Date" system line
# Qwen: <|im_end|> closes every ChatML turn, so it is also the answer's final label.
END_OF_TURN = "<|im_end|>"
# Qwen: the tokenizers already set <|endoftext|> as the pad token, so this is only a
# fallback for configure_tokenizer; it is distinct from <|im_end|>.
PAD_TOKEN = "<|endoftext|>"
# Qwen3: disable thinking mode. Otherwise the generation prompt ends with a bare
# "<|im_start|>assistant\n" and the model learns to write an empty <think></think>
# block before every query. Qwen2.5's template has no thinking mode and ignores this.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

SYSTEM_PROMPT = (
    "You are a text-to-SQL system.\n"
    "Use ONLY tables/columns from the schema.\n"
    "Return exactly ONE SQLite SQL query and nothing else.\n"
    "Do NOT include explanations, comments, code fences, or the database id."
)


def configure_tokenizer(tokenizer, padding_side: str):
    """Make sure the tokenizer has a dedicated pad token and set the padding side.

    The pad token must not be the end-of-turn token (the usual `pad = eos`
    shortcut): training masks pad positions out of the loss, which would also mask
    the END_OF_TURN token (`<|im_end|>` for Qwen) the model has to learn to emit
    after the SQL."""
    if tokenizer.pad_token is None:
        if PAD_TOKEN not in tokenizer.get_vocab():
            raise ValueError(f"{PAD_TOKEN} is not in the tokenizer vocabulary; pick another unused pad token")
        tokenizer.pad_token = PAD_TOKEN
    if tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids(END_OF_TURN):
        raise ValueError(f"pad token must differ from {END_OF_TURN}, or the end of every answer is masked from the loss")
    tokenizer.padding_side = padding_side
    return tokenizer


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
