"""## Multi-GPU QLoRA fine-tuning with FSDP (for models too large for one GPU)

Same pipeline as finetune_qlora.py (same data, prompts, loss masking, LoRA and training
hyper-parameters, early stopping, adapter merge), but the 4-bit model is sharded across
several GPUs with FSDP (Fully Sharded Data Parallel), so every GPU holds only a slice of
the weights and all GPUs train in parallel. This is what makes e.g. Llama 3.3 70B
(~37GB in 4-bit) trainable on 24GB GPUs.

Launch it with accelerate, one process per GPU (see finetune_qlora_fsdp_job.sh):

    accelerate launch --config_file fsdp_qlora.yaml --num_processes 4 finetune_qlora_fsdp.py

Everything shared with finetune_qlora.py (helpers and hyper-parameters) is imported from
it; only the settings that must differ for sharded training are defined here.
"""
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import torch
import wandb
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

import finetune_qlora as base
from spider_prompts import configure_tokenizer

# --------------------
# Settings that differ from finetune_qlora.py
# --------------------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_TAG = "70B-fsdp"  # suffix of the run directory / Hub repo name

# Per-GPU batch. Effective batch stays base.EFFECTIVE_BATCH_SIZE: gradient accumulation
# is derived from it and the number of GPUs.
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4

# A 70B eval pass over the whole dev set costs about as much as the 10 training steps
# between evaluations in finetune_qlora.py, so evaluate half as often and on a subset.
# Patience is lowered to match, so early stopping still waits ~60 optimizer steps.
CHECKPOINT_EVAL_STEPS = base.CHECKPOINT_EVAL_STEPS if base.LITE_MODE else 20
EARLY_STOPPING_PATIENCE = 3
EVAL_SAMPLES = 500  # None evaluates on the full dev set

# The paged bitsandbytes optimizer is not supported on FSDP-sharded parameters. The
# LoRA parameters are small, so plain AdamW's state costs little memory.
OPTIMIZER = "adamw_torch"

# Merging needs the bf16 base model in CPU RAM (~140GB for 70B); disable it if the node
# does not have that much, and merge later on a large-memory node.
MERGE_ADAPTER = True

# Rank 0 loads the whole model while the other ranks wait for it, and merging runs on
# rank 0 only; the default 10-30 minute collective timeout is too short for 70B.
DISTRIBUTED_TIMEOUT = timedelta(hours=2)


def main() -> None:
    # Set up the process group before anything else so the long timeout applies
    # (TrainingArguments reuses this shared state).
    state = PartialState(timeout=DISTRIBUTED_TIMEOUT)
    state.print(f"LITE_MODE is {'ON' if base.LITE_MODE else 'OFF'}; {state.num_processes} processes")

    # HF_TOKEN from .env; huggingface_hub reads it from the environment on every rank
    load_dotenv()
    if state.is_main_process:
        login(os.environ["HF_TOKEN"], add_to_git_credential=True)

    # The run name and shuffle seed must be identical on all ranks (same output dir,
    # same data order for the distributed sampler), so rank 0 picks them.
    shared = [
        f"{datetime.now():%Y-%m-%d_%H.%M.%S}-finetune-QLORA" + ("-lite" if base.LITE_MODE else ""),
        base.SHUFFLE_SEED if base.SHUFFLE_SEED is not None else random.randrange(2**32),
    ]
    broadcast_object_list(shared, from_process=0)
    run_name, shuffle_seed = shared
    project_run_name = f"{base.PROJECT_NAME}-{run_name}-{MODEL_TAG}"

    # Weights & Biases, on the main process only (the Trainer only logs from there)
    os.environ["WANDB_PROJECT"] = base.PROJECT_NAME
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(base.OUTPUT_DIR)
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    if state.is_main_process:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project=base.PROJECT_NAME, name=run_name)
        wandb.config.update({"shuffle_seed": shuffle_seed, "model_id": MODEL_ID})

    # Load the dataset
    dataset = DatasetDict({
        "train": Dataset.from_list(base.load_sft_jsonl(base.TRAIN_JSONL)),
        "validation": Dataset.from_list(base.load_sft_jsonl(base.DEV_JSONL)),
    })
    state.print(f"Shuffling datasets with seed {shuffle_seed}")
    dataset = dataset.shuffle(seed=shuffle_seed)
    state.print(f"Loaded {len(dataset['train'])} train / {len(dataset['validation'])} validation examples")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Same 4-bit quantization as finetune_qlora.py. bnb_4bit_quant_storage must equal the
    # model dtype: FSDP can only shard parameters that all have the same dtype.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    # Quantize into CPU RAM; FSDP then moves one shard to each GPU when training starts.
    # Without an explicit device_map, transformers 5 puts a 4-bit model on the current GPU,
    # so every rank would try to fit the whole model (~37GB for 70B) on its own GPU. Every
    # rank loads its own copy: transformers' rank-0-only loading skips quantized models.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch_dtype, quantization_config=quant_config, device_map={"": "cpu"}
    )
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(MODEL_ID), padding_side="right")

    # prepare_model_for_kbit_training is deliberately not called: it upcasts the
    # non-quantized layers to fp32, and FSDP needs one dtype throughout. What it would do
    # for gradient checkpointing is this: with frozen embeddings, the checkpointed
    # activations need an input that requires grad.
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        lora_alpha=base.LORA_ALPHA,
        lora_dropout=base.LORA_DROPOUT,
        r=base.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=base.TARGET_MODULES,
    )

    gradient_accumulation_steps = max(1, base.EFFECTIVE_BATCH_SIZE // (BATCH_SIZE * state.num_processes))
    state.print(f"Effective batch: {BATCH_SIZE} per GPU x {state.num_processes} GPUs x "
                f"{gradient_accumulation_steps} accumulation steps")

    train_parameters = SFTConfig(
        output_dir=str(base.OUTPUT_DIR / project_run_name),
        num_train_epochs=base.EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        # Recompute activations in the backward pass instead of storing them; reentrant
        # checkpointing is the variant that works with FSDP + QLoRA.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim=OPTIMIZER,
        weight_decay=base.WEIGHT_DECAY,
        logging_steps=base.LOGGING_STEPS,
        save_strategy=base.SAVE_STRATEGY,
        eval_strategy="steps",
        eval_steps=CHECKPOINT_EVAL_STEPS,
        learning_rate=base.LEARNING_RATE,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_steps=base.WARMUP_RATIO,
        lr_scheduler_type=base.LR_SCHEDULER_TYPE,
        push_to_hub=not base.LITE_MODE,
        report_to="wandb",
        run_name=run_name,
        max_length=base.MAX_SEQUENCE_LENGTH,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        load_best_model_at_end=not base.LITE_MODE,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        ddp_timeout=int(DISTRIBUTED_TIMEOUT.total_seconds()),
    )

    # Drop examples that do not fit max_length, as in finetune_qlora.py
    token_length = base.build_token_length_fn(tokenizer)
    split_caps = {
        "train": base.LITE_TRAIN_SAMPLES if base.LITE_MODE else None,
        "validation": base.LITE_EVAL_SAMPLES if base.LITE_MODE else EVAL_SAMPLES,
    }
    for split in dataset:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda example: token_length(example) <= train_parameters.max_length
        )
        state.print(f"{split}: dropped {before - len(dataset[split])}/{before} examples "
                    f"exceeding max_length={train_parameters.max_length} tokens")
        if not len(dataset[split]):
            raise ValueError(
                f"All {before} {split} examples exceed max_length={train_parameters.max_length} tokens. "
                "Increase MAX_SEQUENCE_LENGTH in finetune_qlora.py."
            )
        cap = split_caps[split]
        if cap is not None:
            dataset[split] = dataset[split].select(range(min(cap, len(dataset[split]))))
            state.print(f"Trimmed {split} to {len(dataset[split])} examples (post-filter)")

    callbacks = [] if base.LITE_MODE else [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        args=train_parameters,
        processing_class=tokenizer,
        data_collator=base.build_collate_fn(tokenizer, train_parameters.max_length),
        callbacks=callbacks,
    )

    if trainer.is_fsdp_enabled:
        # Wrap the LoRA layers in their own FSDP units: an FSDP unit can't mix trainable
        # (LoRA) and frozen (base) parameters.
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
    else:
        state.print("WARNING: FSDP is not enabled; launch with "
                    "`accelerate launch --config_file fsdp_qlora.yaml ...`")
    if hasattr(trainer.model, "print_trainable_parameters") and state.is_main_process:
        trainer.model.print_trainable_parameters()

    trainer.train()

    # Checkpoints are saved sharded during training; gather the adapter into one normal
    # adapter_model.safetensors for the final save (collective: runs on every rank).
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    if state.is_main_process:
        wandb.finish()

    del model
    del trainer
    torch.cuda.empty_cache()
    state.wait_for_everyone()
    # The merge runs on rank 0 only; the other ranks are done
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not state.is_main_process or not MERGE_ADAPTER:
        return

    # Merge the adapter into the bf16 base model on the CPU, as in finetune_qlora.py
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    merged_model_dir = Path(train_parameters.output_dir) / base.MERGED_SUBDIR
    peft_model = PeftModel.from_pretrained(model, train_parameters.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(merged_model_dir)
    print(f"Merged model saved to {merged_model_dir}")

    if not base.LITE_MODE:
        base.update_latest_link(base.LATEST_MODEL_LINK, merged_model_dir)


if __name__ == "__main__":
    main()
