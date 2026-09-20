"""## Test Model Inference and generate SQL queries

After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.

Note: Evaluating generative AI models is not a trivial task since one input can have multiple correct outputs. This guide only focuses on manual evaluation and vibe checks.
"""
from pathlib import Path
from typing import Any
from transformers import AutoModelForMultimodalLM, AutoProcessor
from random import randint
from transformers import pipeline, GenerationConfig, pipeline
import json

from spider_prompts import build_prompt_messages, load_tables, schema_to_text

DATA_DIR = Path(__file__).parent / "spider_data"
TEST_TABLES_JSON = DATA_DIR / "test_tables.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider-format JSON file into a list of example records."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_test_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the test split."""
    return load_json(data_dir / "test.json")


test_samples = load_test_dataset()
test_tables = load_tables(TEST_TABLES_JSON)

model_id = str(Path(__file__).parent / "out" / "text2sql_qlora")

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
test_sample = test_samples[randint(0, len(test_samples) - 1)]

# Convert the test example into the prompt the model was trained on, rendered with the Gemma template
messages = build_prompt_messages(test_sample["question"], test_sample["db_id"], test_tables)
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# Generate our SQL query.
outputs = pipe(text_inputs=prompt, generation_config=config)

print(f"DB ID: ", test_sample["db_id"])
print(f"Schema:\n", schema_to_text(test_tables[test_sample["db_id"]]))
print(f"Question:\n", test_sample["question"])
print(f"Original Answer:\n{test_sample['query']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].split('<turn|>')[0].strip()}")
