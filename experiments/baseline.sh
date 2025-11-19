#! /bin/bash
experiment_name="deepseek-baseline"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("gsm8k_test")
temperature=(0.7)
max_tokens=(10000)
num_samples=(1)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    # if the directory results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens exists, skip
                    if [ ! -f "results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/predictions.jsonl" ]; then
                        echo "Running baseline for $model $ds $num_samples $temperature $max_tokens"
                        python scripts/run_baseline.py \
                            --model_name_or_path $model \
                            --dataset $ds \
                            --num_samples $num_samples \
                            --temperature $temperature \
                            --max_tokens $max_tokens \
                            --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens
                    fi
                    
                    if [ ! -f "results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/judgement.jsonl" ]; then
                        echo "Running judgement for $model $ds $num_samples $temperature $max_tokens"
                        python scripts/run_judge.py \
                            --model_name_or_path openai/gpt-oss-120b \
                            --predictions_path results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/predictions.jsonl \
                            --max_tokens 8192 \
                            --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens
                    fi
                done
            done
        done
    done
done