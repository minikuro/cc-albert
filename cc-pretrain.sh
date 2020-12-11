#!/bin/bash

export MODEL_DIR="./base"

export PROCESSED_DATA="./data_preproc"

export MODEL_CONFIG="./model_configs"

export PRETRAINED_MODEL="./pretrained_models"
mkdir -p $PRETRAINED_MODEL


python run_pretraining.py --do_train --strategy_type=one \
  --albert_config_file=${MODEL_CONFIG}/base/config.json \
  --num_train_epochs=3 \
  --train_batch_size=10 \
  --warmup_proportion=0.1 \
  --input_files=${PROCESSED_DATA}/train.tf_record \
  --meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
  --output_dir=${PRETRAINED_MODEL}
#  --init_checkpoint=pretrained_model

  #--output_dir=${PRETRAINED_MODEL} --train_batch_size=8 --num_train_epochs=3
  #--output_dir=${PRETRAINED_MODEL} --strategy_type=mirror --train_batch_size=32 --num_train_epochs=3
