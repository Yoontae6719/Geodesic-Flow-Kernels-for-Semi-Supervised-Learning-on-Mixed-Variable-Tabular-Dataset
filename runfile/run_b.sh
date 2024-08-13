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

  ["Insurance"]="--root_path ./all_dataset/Insurance/ --data Insurance --train_csv train.csv --test_csv test.csv --model_id Insurance --num_cont 3 --cont_emb 33 --num_cat 7 --cat_unique_list 36 2 2 2 9 15 4 --out_class 2 --batch_size 128 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 2"
  ["adult"]="--root_path ./all_dataset/adult/ --data adult --train_csv train.csv --test_csv test.csv --model_id adult --num_cont 6 --cont_emb 16 --num_cat 8 --cat_unique_list 9 16 7 15 6 5 2 42 --out_class 2 --batch_size 256 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 1 --depth 2" 
  ["bank-marketing"]="--root_path ./all_dataset/bank-marketing/ --data bank --train_csv train.csv --test_csv test.csv --model_id bank-marketing --num_cont 7 --cont_emb 14 --num_cat 9 --cat_unique_list 12 3 4 2 2 2 3 12 4 --out_class 2 --batch_size 256 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 1 --depth 2"
  
  ["okcupid_stem"]="--root_path ./all_dataset/okcupid_stem/ --data okcupid_stem --train_csv train.csv --test_csv test.csv --model_id okcupid_stem --num_cont 2 --cont_emb 50 --num_cat 11 --cat_unique_list 12 6 3 32 174 142 3 2 48 5 5 --out_class 3 --batch_size 128 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"

  


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
