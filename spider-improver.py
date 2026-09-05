from dotenv import load_dotenv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import login
import torch
import wandb

# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "Spider Improver"
HF_USER = "ed-donner" # your HF name here!

LITE_MODE = True

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
  RUN_NAME += "-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyper-parameters - overall

EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
MAX_SEQUENCE_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 1

# Hyper-parameters - QLoRA

QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.1

# Hyper-parameters - training

LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking

VAL_SIZE = 500 if LITE_MODE else 1000
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200
LOG_TO_WANDB = True

DATA_DIR = Path(__file__).parent / "spider_data"


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider-format JSON file into a list of example records."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_train_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the training split: train_spider.json concatenated with train_others.json."""
    train_spider = load_json(data_dir / "train_spider.json")
    train_others = load_json(data_dir / "train_others.json")
    return train_spider + train_others


def load_dev_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the dev split."""
    return load_json(data_dir / "dev.json")


def load_test_dataset(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load the test split."""
    return load_json(data_dir / "test.json")


def main() -> None:
    load_dotenv(override=True)
    
    # Log in to HuggingFace

    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)

    # Log in to Weights & Biases
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb.login()

    # Configure Weights & Biases to record against our project
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    # Load the dataset
    train = load_train_dataset()
    dev = load_dev_dataset()
    test = load_test_dataset()

    print(f"Loaded {len(train):,} training examples")
    print(f"Loaded {len(dev):,} dev examples")
    print(f"Loaded {len(test):,} test examples")




if __name__ == "__main__":
    main()
