#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(
  ["fars"]="--root_path ./all_dataset/fars/ --data fars --train_csv train.csv --test_csv test.csv --model_id fars --num_cont 14 --cont_emb 7 --num_cat 15 --cat_unique_list 3 10 4 10 3 4 7 10 4 5 7 7 7 9 3 --out_class 7 --batch_size 256 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"
  ["Diabetes130US"]="--root_path ./all_dataset/Diabetes130US/ --data Diabetes130US --train_csv train.csv --test_csv test.csv --model_id Diabetes130US --num_cont 11 --cont_emb 9 --num_cat 34 --cat_unique_list 6 3 10 10 17 73 711 737 776 4 4 4 4 4 4 4 2 4 4 2 4 4 4 4 2 3 4 4 2 2 2 2 2 2 --out_class 3 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 1"
  ["kick"]="--root_path ./all_dataset/kick/ --data kick --train_csv train.csv --test_csv test.csv --model_id kick --num_cont 14 --cont_emb 7 --num_cat 16 --cat_unique_list 3 32 938 133 811 16 3 3 3 4 12 4 72 148 37 2 --out_class 2 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 1 --depth 2"



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
    for label_ratio in 0.1; do
        for is_noise in 0; do
            run_experiment $dataset $label_ratio $is_noise
    done
  done
done

# Wait for all background processes to finish
wait
