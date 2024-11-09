#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0


# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(
  ["qsar"]="--root_path ./all_dataset/qsar/ --data qsar --train_csv train.csv --test_csv test.csv --model_id qsar --num_cont 30 --cont_emb 3 --num_cat 10 --cat_unique_list 4 4 2 2 4 13 8 4 4 2 --out_class 2 --batch_size 16 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 1 --depth 2"
  ["cmc"]="--root_path ./all_dataset/cmc/ --data cmc --train_csv train.csv --test_csv test.csv --model_id cmc --num_cont 2 --cont_emb 50 --num_cat 7 --cat_unique_list 4 4 2 2 4 4 2 --out_class 3 --batch_size 16 --embed_dim 24 --dim_head 32 --fcn_hidden 12 --heads 4 --depth 4"
  ["jasmine"]="--root_path ./all_dataset/jasmine/ --data jasmine --train_csv train.csv --test_csv test.csv --model_id jasmine --num_cont 8 --cont_emb 12 --num_cat 136 --cat_unique_list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["credit-g"]="--root_path ./all_dataset/credit-g/ --data credit-g --train_csv train.csv --test_csv test.csv --model_id credit-g --num_cont 7 --cont_emb 14 --num_cat 13 --cat_unique_list 4 5 10 5 5 4 3 4 3 3 4 2 2 --out_class 2 --batch_size 16 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 1"
  ["dresses-sales"]="--root_path ./all_dataset/dresses-sales/ --data dresses-sales --train_csv train.csv --test_csv test.csv --model_id dresses-sales --num_cont 1 --cont_emb 100 --num_cat 11 --cat_unique_list 13 8 7 9 17 16 5 24 22 25 15 --out_class 2 --batch_size 8 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["credit-approval"]="--root_path ./all_dataset/credit-approval/ --data credit-approval --train_csv train.csv --test_csv test.csv --model_id credit-approval --num_cont 6 --cont_emb 16 --num_cat 9 --cat_unique_list 2 3 3 14 9 2 2 2 3 --out_class 2 --batch_size 8 --embed_dim 24 --dim_head 32 --fcn_hidden 12 --heads 4 --depth 1"
  ["eye_movements"]="--root_path ./all_dataset/eye_movements/ --data eye_movements --train_csv train.csv --test_csv test.csv --model_id eye_movements --num_cont 20 --cont_emb 5 --num_cat 3 --cat_unique_list 2 2 2 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 2"
  ["churn"]="--root_path ./all_dataset/churn/ --data churn --train_csv train.csv --test_csv test.csv --model_id churn --num_cont 16 --cont_emb 6 --num_cat 4 --cat_unique_list 3 2 2 10 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 2"
  ["Shipping"]="--root_path ./all_dataset/Shipping/ --data Shipping --train_csv train.csv --test_csv test.csv --model_id Shipping --num_cont 5 --cont_emb 20 --num_cat 4 --cat_unique_list 5 3 3 2 --out_class 2 --batch_size 16 --embed_dim 36 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["KDD"]="--root_path ./all_dataset/KDD/ --data KDD --train_csv train.csv --test_csv test.csv --model_id KDD --num_cont 34 --cont_emb 3 --num_cat 11 --cat_unique_list 2 4 4 11 3 5 2 2 7 5 7 --out_class 2 --batch_size 32 --embed_dim 24 --dim_head 32 --fcn_hidden 6 --heads 4 --depth 2"
  ["online_shoppers"]="--root_path ./all_dataset/online_shoppers/ --data online_shoppers --train_csv train.csv --test_csv test.csv --model_id online_shoppers --num_cont 4 --cont_emb 25 --num_cat 13 --cat_unique_list 12 8 3 2823 10 9 287 7806 16 20 1064 2 26 --out_class 2 --batch_size 128 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 2 --depth 1"
  ["shrutime"]="--root_path ./all_dataset/shrutime/ --data shrutime --train_csv train.csv --test_csv test.csv --model_id shrutime --num_cont 4 --cont_emb 25 --num_cat 6 --cat_unique_list 3 2 11 2 2 4 --out_class 2 --batch_size 64 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"
)

# Function to run the experiment
run_experiment() {
  local dataset=$1
  local label_ratio=$2
  local is_noise=$3
  local loss_weight=$4

  python -u run_paper.py \
    ${common_params[@]} \
    ${datasets[$dataset]} \
    --model GKSMT \
    --label_ratio $label_ratio \
    --is_noise $is_noise \
    --loss_weight $loss_weight  &
}

# Run experiments for both datasets
for dataset in "${!datasets[@]}"; do
    for label_ratio in 0.2; do
        for is_noise in 0; do
            for loss_weight in 1.0; do
                run_experiment $dataset $label_ratio $is_noise $loss_weight
        done
    done
  done
done

# Wait for all background processes to finish
wait


# Set CUDA device
export CUDA_VISIBLE_DEVICES=0


# Define common parameters
common_params=(
  --is_training 1
  --lambda_value 0.9
  --cat_mask_ratio 0.2

)

# Define datasets
declare -A datasets=(
  ["qsar"]="--root_path ./all_dataset/qsar/ --data qsar --train_csv train.csv --test_csv test.csv --model_id qsar --num_cont 30 --cont_emb 3 --num_cat 10 --cat_unique_list 4 4 2 2 4 13 8 4 4 2 --out_class 2 --batch_size 16 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 1 --depth 2"
  ["cmc"]="--root_path ./all_dataset/cmc/ --data cmc --train_csv train.csv --test_csv test.csv --model_id cmc --num_cont 2 --cont_emb 50 --num_cat 7 --cat_unique_list 4 4 2 2 4 4 2 --out_class 3 --batch_size 16 --embed_dim 24 --dim_head 32 --fcn_hidden 12 --heads 4 --depth 4"
  ["jasmine"]="--root_path ./all_dataset/jasmine/ --data jasmine --train_csv train.csv --test_csv test.csv --model_id jasmine --num_cont 8 --cont_emb 12 --num_cat 136 --cat_unique_list 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["credit-g"]="--root_path ./all_dataset/credit-g/ --data credit-g --train_csv train.csv --test_csv test.csv --model_id credit-g --num_cont 7 --cont_emb 14 --num_cat 13 --cat_unique_list 4 5 10 5 5 4 3 4 3 3 4 2 2 --out_class 2 --batch_size 16 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 1"
  ["dresses-sales"]="--root_path ./all_dataset/dresses-sales/ --data dresses-sales --train_csv train.csv --test_csv test.csv --model_id dresses-sales --num_cont 1 --cont_emb 100 --num_cat 11 --cat_unique_list 13 8 7 9 17 16 5 24 22 25 15 --out_class 2 --batch_size 8 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["credit-approval"]="--root_path ./all_dataset/credit-approval/ --data credit-approval --train_csv train.csv --test_csv test.csv --model_id credit-approval --num_cont 6 --cont_emb 16 --num_cat 9 --cat_unique_list 2 3 3 14 9 2 2 2 3 --out_class 2 --batch_size 8 --embed_dim 24 --dim_head 32 --fcn_hidden 12 --heads 4 --depth 1"
  ["eye_movements"]="--root_path ./all_dataset/eye_movements/ --data eye_movements --train_csv train.csv --test_csv test.csv --model_id eye_movements --num_cont 20 --cont_emb 5 --num_cat 3 --cat_unique_list 2 2 2 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 2"
  ["churn"]="--root_path ./all_dataset/churn/ --data churn --train_csv train.csv --test_csv test.csv --model_id churn --num_cont 16 --cont_emb 6 --num_cat 4 --cat_unique_list 3 2 2 10 --out_class 2 --batch_size 32 --embed_dim 36 --dim_head 64 --fcn_hidden 12 --heads 4 --depth 2"
  ["Shipping"]="--root_path ./all_dataset/Shipping/ --data Shipping --train_csv train.csv --test_csv test.csv --model_id Shipping --num_cont 5 --cont_emb 20 --num_cat 4 --cat_unique_list 5 3 3 2 --out_class 2 --batch_size 16 --embed_dim 36 --dim_head 64 --fcn_hidden 6 --heads 4 --depth 2"
  ["KDD"]="--root_path ./all_dataset/KDD/ --data KDD --train_csv train.csv --test_csv test.csv --model_id KDD --num_cont 34 --cont_emb 3 --num_cat 11 --cat_unique_list 2 4 4 11 3 5 2 2 7 5 7 --out_class 2 --batch_size 32 --embed_dim 24 --dim_head 32 --fcn_hidden 6 --heads 4 --depth 2"
  ["online_shoppers"]="--root_path ./all_dataset/online_shoppers/ --data online_shoppers --train_csv train.csv --test_csv test.csv --model_id online_shoppers --num_cont 4 --cont_emb 25 --num_cat 13 --cat_unique_list 12 8 3 2823 10 9 287 7806 16 20 1064 2 26 --out_class 2 --batch_size 128 --embed_dim 24 --dim_head 64 --fcn_hidden 12 --heads 2 --depth 1"
  ["shrutime"]="--root_path ./all_dataset/shrutime/ --data shrutime --train_csv train.csv --test_csv test.csv --model_id shrutime --num_cont 4 --cont_emb 25 --num_cat 6 --cat_unique_list 3 2 11 2 2 4 --out_class 2 --batch_size 64 --embed_dim 24 --dim_head 64 --fcn_hidden 6 --heads 2 --depth 2"
)

# Function to run the experiment
run_experiment() {
  local dataset=$1
  local label_ratio=$2
  local is_noise=$3
  local loss_weight=$4

  python -u run_paper.py \
    ${common_params[@]} \
    ${datasets[$dataset]} \
    --model GKSMT \
    --label_ratio $label_ratio \
    --is_noise $is_noise \
    --loss_weight $loss_weight  &
}

# Run experiments for both datasets
for dataset in "${!datasets[@]}"; do
    for label_ratio in 0.2; do
        for is_noise in 1; do
            for loss_weight in 1.0; do
                run_experiment $dataset $label_ratio $is_noise $loss_weight
        done
    done
  done
done
# Wait for all background processes to finish
wait
