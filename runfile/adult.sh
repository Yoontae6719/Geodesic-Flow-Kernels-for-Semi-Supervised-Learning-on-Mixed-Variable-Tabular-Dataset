#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1


# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(

  ["adult"]="--root_path ./all_dataset/adult/ --data adult --train_csv train.csv --test_csv test.csv --model_id adult --num_cont 6 --cont_emb 16 --num_cat 8 --cat_unique_list 9 16 7 15 6 5 2 42 --out_class 2 --batch_size 256 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 1 --depth 2" 


)

# Function to run the experiment
run_experiment() {
  local dataset=$1
  local label_ratio=$2
  local is_noise=$3

  python -u run.py \
    ${common_params[@]} \
    ${datasets[$dataset]} \
    --model GKSMT \
    --label_ratio $label_ratio \
    --is_noise $is_noise &
}

# Run experiments for both datasets
for dataset in "${!datasets[@]}"; do
    for label_ratio in 0.2; do
        for is_noise in 0; do
            run_experiment $dataset $label_ratio $is_noise
    done
  done
done

# Wait for all background processes to finish
wait
