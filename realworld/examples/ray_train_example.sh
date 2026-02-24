#!/bin/bash

# Example script to run a ray training experiment.
# You may wish to tune the resources provided to each Ray worker
# using the --gpu_per_worker and --cpu_per_worker flags, or increase
# workers with the --num_workers flag.

# echo 'activating virtual environment'
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate tableshift
models=("mlp" "group_dro" "irm")

# Loop through each model and run the command.
for model in "${models[@]}"; do
	ulimit -u 127590 && python scripts/ray_train.py \
		--experiment diabetes_readmission \
		--num_samples 100 \
		--models "$model" \
		--gpu_per_worker 0.90

	ulimit -u 127590 && python scripts/ray_train.py \
		--experiment acsfoodstamps \
		--num_samples 100 \
		--models "$model" \
		--gpu_per_worker 0.90

	ulimit -u 127590 && python scripts/ray_train.py \
		--experiment brfss_diabetes \
		--num_samples 100 \
		--models "$model" \
		--gpu_per_worker 0.90

	ulimit -u 127590 && python scripts/ray_train.py \
		--experiment acsincome \
		--num_samples 100 \
		--models "$model" \
		--gpu_per_worker 0.90

	ulimit -u 127590 && python scripts/ray_train.py \
		--experiment acsunemployment \
		--num_samples 100 \
		--models "$model" \
		--gpu_per_worker 0.90
done