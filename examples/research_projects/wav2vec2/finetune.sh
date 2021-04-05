#!/usr/bin/env bash
LANG="lg"
OUTPUT_DIR="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-${LANG}-augment"
CHECKPOINT="$(ls -d ${OUTPUT_DIR})"
if [ -z $CHECKPOINT ]; then
    CHECKPOINT="facebook/wav2vec2-large-xlsr-53"
fi
python run_common_voice.py \
    --model_name_or_path="$CHECKPOINT" \
    --dataset_config_name="$LANG" \
    --overwrite_output_dir="0" \
    --load_processor="1" \
    --augmented="1" \
    --output_dir="$OUTPUT_DIR" \
    --cache_dir=/workspace/raw_data/$LANG \
    --train_split_name="train+validation+other" \
    --eval_split_name="test" \
    --num_train_epochs="60" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --evaluation_strategy="steps" \
    --learning_rate="2e-4" \
    --warmup_steps="1024" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="64" \
    --eval_steps="64" \
    --save_total_limit="1" \
    --logging_steps="64" \
    --group_by_length \
    --feat_proj_dropout="0.05" \
    --attention_dropout="0.1" \
    --activation_dropout="0.05" \
    --hidden_dropout="0.05" \
    --mask_time_prob="0.1" \
    --layerdrop="0.05" \
    --gradient_checkpointing \
    --do_train --do_eval 
#    --max_val_samples="2048" \
