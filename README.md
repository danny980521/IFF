# IFF
Instruction Following Factory: Instruction Following Model의 학습부터 평가까지

## Setup
```
git clone https://github.com/danny980521/IFF.git
cd IFF
git submodule update --init --recursive
bash setup.sh
```

## (Optional) Base Model & Judge Model 다운로드
```
mkdir models && cd models
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/kakaocorp/kanana-nano-2.1b-base
git clone https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
cd ..
```

## SFT Train
trl의 `SFTTrainer`를 사용해 학습하는 코드입니다.
```
deepspeed --num_gpus=8 sft.py \
    --model_path {MODEL_PATH} \
    --save_path {SAVE_PATH} \
    --dataset_path {DATASET_PATH} \
    --dataset_name {DATASET_NAME} \
    --train_batch_size {TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs {NUM_TRAIN_EPOCHS}
```

## Evaluation
`ifeval`의 경우 lm-eval-harness를 이용하며, `mt-bench`의 경우 FastChat을 사용합니다.
`mt-bench`를 평가할 때 judge model을 직접 vLLM을 사용해 GPU에 올려 평가하며, OpenAI API 콜을 원한다면 원본 레포를 사용하면 됩니다.
```
bash eval.sh \
    {MODEL_PATH} \
    {MODEL_ID} \
    {TENSOR_PARALLEL_SIZE} \
    {DATA_PARALLEL_SIZE} \
    {JUDGE_MODEL_PATH} \
    {JUDGE_MODEL_ID} \
    {JUDGE_MODEL_TP_SIZE} \
    {HF_DATASETS_CACHE}
```

## Results
| 모델 이름                                          | ifeval (inst, loose) | mt-bench (single, avg) |
|--------------------------------------------------|----------------------|-------------------------|
| kanana-nano-2.1b-base                            | 0.2614               | 4.50                    |
| kanana-nano-2.1b-instruction                     | **0.7194**           | **5.79**                |
| kanana-nano-2.1b-base-smol-magpie-ultra-1epoch   | 0.2998               | 4.24                    |

