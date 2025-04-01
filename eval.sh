#!/bin/bash

MODEL_PATH="${1:-"models/kanana-nano-2.1b-base"}"
MODEL_ID="${2:-"kanana-nano-2.1b-base"}"
TENSOR_PARALLEL_SIZE="${3:-1}"
DATA_PARALLEL_SIZE="${4:-8}"
JUDGE_MODEL_PATH="${5:-"models/Qwen2.5-32B-Instruct"}"
JUDGE_MODEL_ID="${6:-"Qwen2.5-32B-Instruct"}"
JUDGE_MODEL_TP_SIZE="${7:-8}"
HF_DATASETS_CACHE="${8:-"/data/di-LLM_458/denver.in/IFF/cache"}"

WORLD_SIZE=$((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE))
GPU_DEVICES=$(seq -s, 0 $((WORLD_SIZE - 1)))

# lm-eval-harness 평가 시작
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" lm_eval --model vllm \
    --model_args pretrained="$MODEL_PATH",tensor_parallel_size="$TENSOR_PARALLEL_SIZE",dtype="float16",gpu_memory_utilization=0.8,data_parallel_size="$DATA_PARALLEL_SIZE" \
    --tasks ifeval \
    --batch_size auto \
    --trust_remote_code \
    --output_path ./lm-evaluation-harness/output/ \
    --log_samples

# komt-bench 평가 시작
cd FastChat/fastchat/llm_judge

# 1) 모델 추론
python gen_model_answer.py \
    --model-path "../../../$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --dtype float16 \
    --num-gpus-per-model "$TENSOR_PARALLEL_SIZE" \
    --num-gpus-total "$WORLD_SIZE"

# 2) vllm serve를 백그라운드에서 실행 (백그라운드 로그는 vllm_serve.log 파일에 기록)
LOG_FILE="../../../vllm_serve.log"

vllm serve "../../../$JUDGE_MODEL_PATH" \
    --served-model-name "$JUDGE_MODEL_ID" \
    --port 8000 \
    --tensor-parallel-size "$JUDGE_MODEL_TP_SIZE" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!
echo "vllm serve started with PID: $VLLM_PID, logging to $LOG_FILE"

# 3) 스크립트 종료 시 vllm serve 프로세스를 종료하도록 trap 설정
trap "echo 'Terminating vllm serve with PID: $VLLM_PID'; kill $VLLM_PID" EXIT

# 4) Health check: 서버가 준비되었는지 확인
HEALTH_CHECK_PAYLOAD="{
    \"model\": \"$JUDGE_MODEL_ID\",
    \"prompt\": \"Kakao is a leading company in South Korea, and it is known for \",
    \"max_tokens\": 32,
    \"top_k\": 1
}"
HEALTH_CHECK_URL="http://localhost:8000/v1/completions"

echo "Waiting for vllm server to be ready..."
until curl --silent --fail -H "Content-Type: application/json" -d "$HEALTH_CHECK_PAYLOAD" "$HEALTH_CHECK_URL" > /dev/null; do
    echo "vllm server not ready yet, retrying in 5 seconds..."
    sleep 5
done
echo "vllm server is ready!"

# 5) vllm 이용해 judge model inference
echo | python gen_judgment.py \
    --model-list "$MODEL_ID" \
    --parallel 10 \
    --judge-model "$JUDGE_MODEL_ID"

# 6) judge 결과 확인
python show_result.py \
    --mode single \
    --input-file ./data/mt_bench/model_judgment/"$JUDGE_MODEL_ID"_single.jsonl
