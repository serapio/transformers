#!/usr/bin/env bash
set -e

LANG="rw"
OUTPUT_DIR="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-${LANG}-apos"
CHECKPOINT="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-${LANG}-apos"
#CHECKPOINT="/workspace/checkpoints/${LANG}/wav2vec2-large-xlsr-kinyarwanda"
#CHECKPOINT="facebook/wav2vec2-large-xlsr-53"
SHARD=$1
if [ -z $SHARD ]; then
    SHARD=1
fi
START="$(expr $SHARD \* 81920)"
END="$(expr $SHARD \* 81920 + 81920)"
EPOCHS=$(expr 10 \* $SHARD)
echo "Using shard $SHARD from $START to $END until $EPOCHS epochs"

python run_common_voice.py \
    --model_name_or_path="$CHECKPOINT" \
    --dataset_config_name="$LANG" \
    --load_processor \
    --augmented="0" \
    --output_dir="$OUTPUT_DIR" \
    --cache_dir=/workspace/raw_data/$LANG \
    --train_split_name="train[${START}:${END}]" \
    --eval_split_name="validation[:15%]" \
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
    --max_val_samples="2048" \
    --max_train_samples="32768" \
    --do_train --do_eval \
 ## --overwrite_output_dir
