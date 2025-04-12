# ref1: https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb
# ref2: https://huggingface.co/docs/trl/en/sft_trainer

import argparse

import torch
from datasets import load_dataset, DatasetDict
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import OnlineDPOConfig, OnlineDPOTrainer


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
        "--reward_model_path",
        type=str,
        default="models/kanana-nano-2.1b-base",
        help="Path to the reward model.",
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
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate during training.",
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
        "--use_vllm",
        action="store_true",
        help="Use VLLM for training.",
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
        torch_dtype=torch.bfloat16,
        is_encoder_decoder=False,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        is_encoder_decoder=False,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )
    # model.config.is_encoder_decoder = False
    # ref_model.config.is_encoder_decoder = False

    # reward model 로드
    reward_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.reward_model_path,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.reward_model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )

    # 데이터셋 로드
    if not args.dataset_name:
        args.dataset_name = None

    ds = load_dataset(
        "json",
        data_dir=args.dataset_path,
        cache_dir=args.cache_dir,
        num_proc=32,
    )

    if isinstance(ds, DatasetDict):
        if "test" not in ds:
            split_dataset = ds["train"].train_test_split(
                test_size=0.01, shuffle=True, seed=42
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = ds["train"]
            eval_dataset = ds["test"]
    else:
        split_dataset = ds.train_test_split(
            test_size=0.01, shuffle=True, seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    # change column name "question" to "prompt"
    # https://github.com/huggingface/trl/blob/5a0cebc7869b0435b4e916a104e0b7b14a7f03f3/trl/data_utils.py#L118-L138
    train_dataset = train_dataset.rename_column("question", "prompt")
    eval_dataset = eval_dataset.rename_column("question", "prompt")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # # Deepspeed Stage 3 (Zero-3) 설정
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "zero_optimization": {
            # "stage": 1,
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "logging_config": {"log_rank_0_only": True},
    }

    # OnlineDPOConfig 설정
    dpo_config = OnlineDPOConfig(
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
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_sequence_length,
        deepspeed=deepspeed_config,
        use_vllm=args.use_vllm,
        dataset_num_proc=32,
        report_to="tensorboard",
        logging_dir=args.logging_dir,
    )

    # OnlineDPOTrainer 초기화
    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # dpo processing
        reward_processing_class=reward_tokenizer, # reward processing
    )

    # 모델 학습
    trainer.train()


if __name__ == "__main__":
    main()
