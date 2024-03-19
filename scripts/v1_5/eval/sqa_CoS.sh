#!/bin/bash

CKPT="/PATH/TO/CKPT"
CKPT_output="llava-v1.5_output"
CKPT_result="llava-v1.5_result"

python -m llava.eval.model_vqa_science_crop \
    --model-path checkpoints/$CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$CKPT_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$CKPT_result.json
