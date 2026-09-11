"""## Test Model Inference and generate SQL queries

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

Note: Evaluating generative AI models is not a trivial task since one input can have multiple correct outputs. This guide only focuses on manual evaluation and vibe checks.
"""
from pathlib import Path
from typing import Any
from transformers import AutoModelForMultimodalLM, AutoProcessor
from random import randint
import re
import sqlite3
from transformers import pipeline, GenerationConfig, pipeline
from datasets import Dataset, DatasetDict
import json

DATA_DIR = Path(__file__).parent / "spider_data"
TEST_DB_DIR = DATA_DIR / "test_database"

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

def get_schema(db_id: str, db_dir: Path = TEST_DB_DIR) -> str:
    """Return the CREATE TABLE statements for a Spider database as schema context."""
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    return "\n\n".join(row[0] for row in rows)


def create_conversation(sample, idx):
    user_content = user_prompt.format(
        context=get_schema(sample["db_id"]),
        question=sample["question"],
    )
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["query"]},
        ]
    }

def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider-format JSON file into a list of example records."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_test_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the test split."""
    return load_json(data_dir / "test.json")


def to_conversations(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce raw Spider records (with their heterogeneous 'sql' parse trees)
    down to just the {question, query} fields the model actually trains on,
    before they ever reach Arrow/pyarrow schema inference."""
    return [create_conversation(sample, idx) for idx, sample in enumerate(samples)]

dataset = DatasetDict({
    "test": Dataset.from_list(to_conversations(load_test_dataset())),
})

model_id = "merged_model"

# Load Model with PEFT adapter
model = AutoModelForMultimodalLM.from_pretrained(
  model_id,
  device_map="auto",
  dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

"""Let's load a random sample from the test dataset and generate a SQL command."""

config = GenerationConfig.from_pretrained(model_id)
config.max_new_tokens = 256
config.eos_token_id = [processor.tokenizer.convert_tokens_to_ids("<turn|>")]

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = processor.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)
print(prompt)

# Generate our SQL query.
outputs = pipe(text_inputs=prompt, generation_config=config)

# Extract the user query and original answer
content1 = test_sample['messages'][1]['content']
content2 = test_sample['messages'][2]['content']

print("content1=", content1)
print("content2=", content2)

print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', content1, re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', content1, re.DOTALL).group(1).strip())
print(f"Original Answer:\n{content2}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip().removesuffix('<turn|>')}")