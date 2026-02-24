# When Shift Happens - Confounding is to Blame
**********************************************
Steps to reproduce synthetic dataset results.
**********************************************

To run the experiments on synthetic data, install the dependices: xgboost, npeet, scikitlearn along with basic modules such as numpy, pandas, and matplotlib.

1. Change directory to code/synthetic
2. To reproduce the results in Figure 2 of the main paper, run: python introfigure.py, , resulting plot will be saved to code/synthetic/results.
3. To reproduce the results in Figure 5 of the main paper, run: python varyingS.py, resulting plot will be saved to code/synthetic/results.
4. To reproduce the results in Figure D14 of the appendix, run: python varyingnoise.py, resulting plot will be saved to code/synthetic/results.
5. To reproduce the results in Figure D15 of the appendix, run: python varyingU.py, resulting plot will be saved to code/synthetic/results.
6. To reproduce the results in Figures D16 - D19 of the appendix, 
   run: python synthetic_xgboost.py and python synthetic_xgboost.py by toggling the variable named 'which' to run experiments for low and high overlapping confounding support.
   Resulting plots will be saved to code/synthetic/results.
7. To reproduce the results in Figures D20-D22, see the instructions provided in code/realworld.


**********************************************
Steps to reproduce real-world dataset results.
**********************************************

To run the experiments on real-world datasets, setup the runtime environment following the instructions at https://github.com/socialfoundations/causal-features/tree/main. Additionally install npeet library for mutual information estimation.

1. Change directory to code/realworld
2. To reproduce the results from Table 2, cd into experiments_causal and run: python confoundingshiftmeasures_data.py
3. To reproduce the resulst from Figure 4, Table 3, Figures C2, C3, C6-C15  run bash examples/script.sh by uncommenting the respective script. Create a JSON with results. Readymade results are already present in 
   results/hyperparamsetting To visualize the results, run: python shiftmeasuresplots_realworld_hyperparametersetting.py 
4. To perform hyperparameter tuning, run: bash examples/ray_train_example.sh. Save the 
   best hyperparameters to examples/hps.py (this step is already performed and best hyperparameters are saved).
   To reproduce the resulst from Figures C4, C5, and Table C2, run bash examples/script.sh by uncommenting the respective script. Create a JSON with results. Readymade results are already present in 
   results/hyperparamsetting_besthps. To visualize the results, run: python shiftmeasuresplots_realworld_best_hyperparametersetting.py 
5. To reproduce the results in D22-D24, run: python plotresults_pmaood.py and python plotresults_rho.py

