# Geodesic Kernel flow for Semi-supervised learning on a Mixed-variable Tabular dataset

This is the origin Pytorch implementation of *Geodesic Kernel flow for Semi-supervised learning on a Mixed-variable Tabular dataset*.

**Warning** This code is not cleand. We will be update the code soon.

**News**(Jun, 01, 2024): The code has been released..
 

## Get Started

To install all dependencies:
```
pip install -r requirements.txt
```

## Dataset
You can access the well pre-processed datasets from Google Drive, then place the downloaded contents under ./all_dataset

## Quick Demo
0. If you want to run the Semi setting, please path to `cd GKP_Semi`. 
1. Download datasets and place them under `./all_dataset`
2. We provide all experiment scripts for demonstration purpose under the folder `./runfile`. For example, you can evaluate on churn and adult datasets by:

```bash
bash ./runfile/churn.sh 
```
```bash
bash ./runfile/adult.sh 
```

3. If you want to experiment with all datasets, run the bash file from run_a to run_d.
```bash
bash ./runfile/run_a.sh
bash ./runfile/run_b.sh
bash ./runfile/run_c.sh
bash ./runfile/run_d.sh 
```

## Detailed usage
Please refer to run.py for the detailed description of each hyperparameter.
We also provide a model where the linear projection of the VSN layer is replaced by KAN. We'll update with more details as they become available.

## Citation
If you find this repo useful, please cite our paper. 

```
Will be update
```

