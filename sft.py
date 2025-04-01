# ref1: https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb
# ref2: https://huggingface.co/docs/trl/en/sft_trainer

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train model with SFT using Deepspeed."
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
        default="HuggingFaceTB/smoltalk",
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
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for model and dataset.",
    )

    args = parser.parse_args()

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=args.cache_dir,
    )

    # 데이터셋 로드
    if not args.dataset_name:
        args.dataset_name = None

    ds = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        cache_dir=args.cache_dir,
    )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

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

    # SFTTrainer 설정
    sft_config = SFTConfig(
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
        deepspeed=deepspeed_config,
    )

    # SFTTrainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 모델 학습
    trainer.train()


if __name__ == "__main__":
    main()
