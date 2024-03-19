#!/bin/bash

CKPT="/PATH/TO/CKPT"

python -m llava.eval.model_vqa_CoS \
    --model-path checkpoints/$CKPT \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$CKPT.json

