BASE_PATH=/data/ib-a100-cluster-a-pri-lmt_967/users/wavy
# BASE_PATH=/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/wavy
IFF_PATH=$BASE_PATH/workspace/study/alignment/IFF

STEPS=1693

deepspeed --num_gpus=8 $IFF_PATH/dpo.py \
    --model_path $IFF_PATH/checkpoints/sft/checkpoint-$STEPS \
    --save_path $IFF_PATH/checkpoints/dpo-from-$STEPS-steps-merged \
    --dataset_path $IFF_PATH/dpo-dataset \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    --save_steps 100 \
    --eval_steps 100 \
    --use_flash_attention \
    --cache_dir $IFF_PATH/cache \
    --logging_dir $IFF_PATH/logs/dpo-from-$STEPS-steps-merged \