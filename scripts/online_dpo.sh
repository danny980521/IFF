BASE_PATH=/data/ib-a100-cluster-a-pri-lmt_967/users/wavy
# BASE_PATH=/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/wavy
IFF_PATH=$BASE_PATH/workspace/study/alignment/IFF

STEPS=1693

torchrun --nnodes 1 --nproc_per_node 8 $IFF_PATH/online_dpo.py \
    --model_path $IFF_PATH/checkpoints/sft/checkpoint-$STEPS \
    --reward_model_path $BASE_PATH/workspace/study/models/GRM-Llama3.2-3B-rewardmodel-ft \
    --save_path $IFF_PATH/checkpoints/online-dpo-from-$STEPS-steps-vllm \
    --dataset_path $BASE_PATH/workspace/study/data/orca_dpo_pairs \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --save_steps 100 \
    --eval_steps 100 \
    --max_new_tokens 2048 \
    --max_sequence_length 4096 \
    --use_flash_attention \
    --use_vllm \
    --cache_dir $IFF_PATH/cache \
    --logging_dir $IFF_PATH/logs/dpo-from-$STEPS-steps-vllm \