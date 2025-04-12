BASE_PATH=/data/ib-a100-cluster-a-pri-lmt_967/users/wavy
# BASE_PATH=/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/wavy
IFF_PATH=$BASE_PATH/workspace/study/alignment/IFF

deepspeed --num_gpus=8 $IFF_PATH/sft.py \
    --model_path $IFF_PATH/models/kanana-nano-2.1b-base \
    --save_path $IFF_PATH/checkpoints/sft \
    --dataset_path $BASE_PATH/workspace/study/data/smoltalk/data/all \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_steps 500 \
    --eval_steps 500 \
    --max_sequence_length 4096 \
    --use_flash_attention \
    --cache_dir $IFF_PATH/cache \
    --logging_dir $IFF_PATH/logs/sft \