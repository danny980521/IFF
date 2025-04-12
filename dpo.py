# ref1: https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb
# ref2: https://huggingface.co/docs/trl/en/sft_trainer

import argparse

from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train model with DPO using Deepspeed."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/kanana-nano-2.1b-base",
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./sft_output",
        help="Save Path to the SFT trained model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./smol-magpie-ultra",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="smol-magpie-ultra",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for training."
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Interval steps for logging."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Interval steps for saving checkpoints.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy (e.g., 'steps').",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Interval steps for evaluation."
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use flash attention for training.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for model and dataset.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for storing logs.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (default: -1).",
    )

    args = parser.parse_args()

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )

    # 데이터셋 로드
    ds = load_from_disk(
        args.dataset_path,
        # cache_dir=args.cache_dir,
    )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Deepspeed Stage 3 (Zero-3) 설정
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "logging_config": {"log_rank_0_only": True},
    }

    # DPOTrainer 설정
    dpo_config = DPOConfig(
        output_dir=args.save_path,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        bf16=True,
        warmup_steps=100,
        deepspeed=deepspeed_config,
        dataset_num_proc=32,
        report_to="tensorboard",
        logging_dir=args.logging_dir,
    )

    # DPOTrainer 초기화
    trainer = DPOTrainer(
        model,
        ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # dpo processing
    )

    # 모델 학습
    trainer.train()


if __name__ == "__main__":
    main()
