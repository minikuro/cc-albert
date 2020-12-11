#!/bin/bash

export SPM_DIR="./spm"

export DATA_DIR="./data"

export PROCESSED_DATA="./data_preproc"

mkdir -p $PROCESSED_DATA

python create_pretraining_data.py --spm_model_file=${SPM_DIR}/spm_sample.model \
  --input_file=${DATA_DIR}/*.txt \
  --output_file=${PROCESSED_DATA}/train.tf_record \
  --meta_data_file_path=${PROCESSED_DATA}/train_meta_data

## tf_records and metadata files will stored in the `PROCESSED_DATA`
