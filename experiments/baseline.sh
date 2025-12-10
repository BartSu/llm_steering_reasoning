#! /bin/bash
experiment_name="qwen3-baseline"
model_name=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-30B-A3B")
dataset=("aime25" "aime24" "MATH500" "AMO-Bench" "minervamath" "olympiadbench")
temperature=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2)
max_tokens=(10000)
num_samples=(100)

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
                    
                    # if [ ! -f "results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/judgement.jsonl" ]; then
                    #     echo "Running judgement for $model $ds $num_samples $temperature $max_tokens"
                    #     python scripts/run_judge.py \
                    #         --model_name_or_path openai/gpt-oss-120b \
                    #         --num_samples 1 \
                    #         --predictions_path results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/predictions.jsonl \
                    #         --max_tokens 8192 \
                    #         --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/
                    # fi
                done
            done
        done
    done
done