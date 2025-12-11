experiment_name="qwen3-4b-aime25-emotion-prompts"
model_name="Qwen/Qwen3-4B"

# all the left folders in the results folder, need go the deepest folder
leaf_folders=$(find results/${experiment_name}/ -type d -links 2 | sort -V)
echo ${leaf_folders}
total_num_of_leaf_folders=$(echo ${leaf_folders} | wc -w)

# print the number of leaf folders
echo "Total number of leaf folders: ${total_num_of_leaf_folders}"
# print the leaf folders
echo "Leaf folders: ${leaf_folders}"


# start calculating metrics
echo "--------------------------------"
echo "Start calculating metrics"
current_num_of_leaf_folders=1
for leaf_folder in ${leaf_folders}; do
    predictions_file="${leaf_folder}/predictions.jsonl"
    metric_path="${leaf_folder}/metrics.json"
    if [ ! -f ${predictions_file} ]; then
        continue
    fi
    if [ -f ${metric_path} ]; then
        continue
    fi
    echo "Calculating metrics for ${predictions_file} (${current_num_of_leaf_folders}/${total_num_of_leaf_folders})"
    python3 analysis/calculate_metric.py --model_name ${model_name} --predictions_file ${predictions_file}
    current_num_of_leaf_folders=$((current_num_of_leaf_folders + 1))
done
