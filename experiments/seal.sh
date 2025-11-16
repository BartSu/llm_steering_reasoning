#!/bin/bash
experiment_name="deepseek-seal"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("gsm8k_test")
temperature=(0.7)
max_tokens=(16384)
num_samples=(1)
vector_scale=(1)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for temp in ${temperature[@]}; do
            for tokens in ${max_tokens[@]}; do
                for samples in ${num_samples[@]}; do
                    for scale in ${vector_scale[@]}; do
                        python scripts/run_seal.py \
                            --model_name_or_path $model \
                            --dataset $ds \
                            --temperature $temp \
                            --max_tokens $tokens \
                            --num_samples $samples \
                            --vector_dir ./vectors/seal/$model \
                            --scale $scale \
                            --save_dir results/$experiment_name/$model/$ds/$temp/$tokens/$samples/$scale
                    done
                done
            done
        done
    done
done