#!/bin/bash

ALBERT_HOME="/content"

spm_train --input=${ALBERT_HOME}/data/sample.txt --model_prefix=spm_sample --vocab_size=100 --character_coverage=0.9995 --model_type=unigram
