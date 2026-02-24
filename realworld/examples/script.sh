#!/bin/bash

# For the experiments in Figure C4, C5, Table C2

# models=("mlp" "group_dro" "irm")
# for model in "${models[@]}"; do
#     echo "Running experiment for model: $model"
#     python examples/run_expt_shiftmeasures.py --experiment diabetes_readmission --model "$model"
#     python examples/run_expt_shiftmeasures.py --experiment acsfoodstamps --model "$model"
#     python examples/run_expt_shiftmeasures.py --experiment brfss_diabetes --model "$model"
#     python examples/run_expt_shiftmeasures.py --experiment acsincome --model "$model"
#     python examples/run_expt_shiftmeasures.py --experiment acsunemployment --model "$model"
# done

# For the experiments in Figure 4, Table 3
# models=("xgb" "mlp" "group_dro" "irm" "vrex")
# for model in "${models[@]}"; do
#     echo "Running experiment for model: $model"
#         python examples/run_expt_shiftmeasures.py --experiment diabetes_readmission --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment acsfoodstamps --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment brfss_diabetes --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment acsincome --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment acsunemployment --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment assistments --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment acspubcov --model "$model"
#         python examples/run_expt_shiftmeasures.py --experiment brfss_blood_pressure --model "$model"
# done
    
