task_name=llava15_7b_DPO
exp_name=llava15_dmdpo

#!/bin/bash

deepspeed --master_port 25433 --include  localhost:0,1,2,7\
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path LLaVA-1.5-7b \
    --data_path your_data_path \
    --image_folder your_image_path \
    --vision_tower your_clip_path \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir your_output_dir \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_dir .ckpt/$task_name-$exp_name/log \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $exp_name \
    --dataloader_num_workers 16 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta 0.1 
