#!/usr/bin/env bash
LANG="rw"
OUTPUT_DIR="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-${LANG}"
CHECKPOINT=  # "$(ls -d ${OUTPUT_DIR})"
if [ -z $CHECKPOINT ]; then
    CHECKPOINT="facebook/wav2vec2-large-xlsr-53"
fi
python run_common_voice.py \
    --model_name_or_path="$CHECKPOINT" \
    --dataset_config_name="$LANG" \
    --overwrite_output_dir \
    --load_processor="0" \
    --augmented="0" \
    --output_dir="$OUTPUT_DIR" \
    --cache_dir=/workspace/raw_data/$LANG \
    --train_split_name="train[:3%]" \
    --eval_split_name="validation[:20%]" \
    --preprocessing_num_workers="4" \
    --num_train_epochs="20" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --evaluation_strategy="steps" \
    --learning_rate="2e-4" \
    --warmup_steps="1024" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="256" \
    --eval_steps="256" \
    --logging_steps="256" \
    --save_total_limit="1" \
    --group_by_length \
    --feat_proj_dropout="0.05" \
    --attention_dropout="0.1" \
    --activation_dropout="0.05" \
    --hidden_dropout="0.05" \
    --mask_time_prob="0.1" \
    --layerdrop="0.05" \
    --gradient_checkpointing \
    --max_val_samples="1024" \
    --do_train --do_eval 
