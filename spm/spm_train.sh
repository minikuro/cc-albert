#!/bin/bash

INPUT_FILE=$1

spm_train --input=${INPUT_FILE} --model_prefix=spm_sample --vocab_size=100 --character_coverage=0.9995 --model_type=unigram
