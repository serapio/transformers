#!/usr/bin/env bash
set -e

LANG="rw"
OUTPUT_DIR="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-${LANG}"
CHECKPOINT="$(ls -d ${OUTPUT_DIR})"
if [ -z $CHECKPOINT ]; then
    CHECKPOINT="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-kinyarwanda"
fi
SHARD=$1
if [ -z $SHARD ]; then
    SHARD=0
fi
START=$(expr $SHARD \* 16384)
END=$(expr $START + 16384)
EPOCHS=$(expr 10 \* $SHARD + 10)
echo "Using shard $SHARD from $START to $END until $EPOCHS epochs"

python run_common_voice.py \
    --model_name_or_path="$CHECKPOINT" \
    --dataset_config_name="$LANG" \
    --load_processor="1" \
    --augmented="0" \
    --output_dir="$OUTPUT_DIR" \
    --cache_dir=/workspace/raw_data/$LANG \
    --train_split_name="train[${START}:${END}]" \
    --eval_split_name="validation[:10%]" \
    --preprocessing_num_workers="8" \
    --num_train_epochs="$EPOCHS" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --evaluation_strategy="steps" \
    --learning_rate="2e-4" \
    --warmup_steps="1024" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="128" \
    --eval_steps="128" \
    --logging_steps="128" \
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
    --max_train_samples="12288" \
    --do_train --do_eval \
   # --overwrite_output_dir 
