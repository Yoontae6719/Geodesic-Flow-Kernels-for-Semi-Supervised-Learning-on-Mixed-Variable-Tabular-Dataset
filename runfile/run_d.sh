#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=3



# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(
  ["nomao"]="--root_path ./all_dataset/nomao/ --data nomao --train_csv train.csv --test_csv test.csv --model_id nomao --num_cont 89 --cont_emb 2 --num_cat 29 --cat_unique_list 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --out_class 2 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"
  ["road-safety"]="--root_path ./all_dataset/road-safety/ --data road-safety --train_csv train.csv --test_csv test.csv --model_id road-safety --num_cont 29 --cont_emb 3 --num_cat 3 --cat_unique_list 2 2 2 --out_class 2 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 1"


)

# Function to run the experiment
run_experiment() {
  local dataset=$1
  local label_ratio=$2
  local is_noise=$3

  python -u run_paper.py \
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

#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=3



# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(
  ["nomao"]="--root_path ./all_dataset/nomao/ --data nomao --train_csv train.csv --test_csv test.csv --model_id nomao --num_cont 89 --cont_emb 2 --num_cat 29 --cat_unique_list 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --out_class 2 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"
  ["road-safety"]="--root_path ./all_dataset/road-safety/ --data road-safety --train_csv train.csv --test_csv test.csv --model_id road-safety --num_cont 29 --cont_emb 3 --num_cat 3 --cat_unique_list 2 2 2 --out_class 2 --batch_size 512 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 1"


)

# Function to run the experiment
run_experiment() {
  local dataset=$1
  local label_ratio=$2
  local is_noise=$3

  python -u run_paper.py \
    ${common_params[@]} \
    ${datasets[$dataset]} \
    --model GKSMT \
    --label_ratio $label_ratio \
    --is_noise $is_noise &
}

# Run experiments for both datasets
for dataset in "${!datasets[@]}"; do
    for label_ratio in 0.2; do
        for is_noise in 1; do
            run_experiment $dataset $label_ratio $is_noise
    done
  done
done

# Wait for all background processes to finish
wait
