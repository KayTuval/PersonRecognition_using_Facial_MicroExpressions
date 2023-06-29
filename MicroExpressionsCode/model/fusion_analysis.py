import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth = 1000


path = "/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/Fusion/2022-07-08_11:00/accuracy_scores.csv"

df = pd.read_csv(path)

basic_models = [
    'LGBP_accuracy',
    'SlowFast_0_accuracy',
    'SlowFast_1_accuracy'
]

fusions = [
    'CosineSimilarityCombiner_accuracy',
    'MacroMajorityVoteCombiner_accuracy',
    'MicroMajorityVoteCombiner_accuracy',
    'SimpleAverageCombiner_accuracy'
]

df["Improvement"] = df[fusions].max(axis=1) - df[basic_models].max(axis=1)

df_of_best = df[(df["Improvement"] > 0) & (df["LGBP_accuracy"] > 0.7)]

df_of_best.sort_values(by=["Improvement"], ascending=False)

df_of_best_sorted = df_of_best.sort_values(by=["Improvement"], ascending=False)

df_of_best_sorted.iloc[0]

"""
LGBP_name                                                                                               C_100.0__kernel_sigmoid__gamma_scale__degree_3
SlowFast_0_name                       SLOWFAST-ALPHA_4__SLOWFAST-BETA_INV_32__TRAIN-BATCH_SIZE_16__SOLVER-BASE_LR_0-001__SOLVER-OPTIMIZING_METHOD_adam
SlowFast_1_name                         SLOWFAST-ALPHA_4__SLOWFAST-BETA_INV_8__TRAIN-BATCH_SIZE_16__SOLVER-BASE_LR_0-001__SOLVER-OPTIMIZING_METHOD_sgd
LGBP_accuracy                                                                                                                                    0.844
SlowFast_0_accuracy                                                                                                                              0.831
SlowFast_1_accuracy                                                                                                                              0.532
Number_Of_Winning_Fusions                                                                                                                            2
CosineSimilarityCombiner_accuracy                                                                                                                 0.87
MacroMajorityVoteCombiner_accuracy                                                                                                               0.935
MicroMajorityVoteCombiner_accuracy                                                                                                               0.766
SimpleAverageCombiner_accuracy                                                                                                                   0.766
Improvement                                                                                                                                      0.091
"""

best_lgbp = df_of_best_sorted.iloc[0]["LGBP_name"]
best_slowfast_0 = df_of_best_sorted.iloc[0]["SlowFast_0_name"]
best_slowfast_1 = df_of_best_sorted.iloc[0]["SlowFast_1_name"]

df_lgbp_and_slowfast_of_best = df[(df["LGBP_name"] == best_lgbp) & (df["SlowFast_0_name"] == best_slowfast_0) & (df["SlowFast_1_name"].isna())]

df_lgbp_and_slowfast_of_best.iloc[0]

"""
LGBP_name                                                                                               C_100.0__kernel_sigmoid__gamma_scale__degree_3
SlowFast_0_name                       SLOWFAST-ALPHA_4__SLOWFAST-BETA_INV_32__TRAIN-BATCH_SIZE_16__SOLVER-BASE_LR_0-001__SOLVER-OPTIMIZING_METHOD_adam
SlowFast_1_name                                                                                                                                    NaN
LGBP_accuracy                                                                                                                                    0.844
SlowFast_0_accuracy                                                                                                                              0.831
SlowFast_1_accuracy                                                                                                                                NaN
Number_Of_Winning_Fusions                                                                                                                            1
CosineSimilarityCombiner_accuracy                                                                                                                0.844
MacroMajorityVoteCombiner_accuracy                                                                                                               0.909
MicroMajorityVoteCombiner_accuracy                                                                                                               0.714
SimpleAverageCombiner_accuracy                                                                                                                   0.714
Improvement                                                                                                                                      0.065
"""


"""
As we can see
for LGBP: C_100.0__kernel_sigmoid__gamma_scale__degree_3
and SF = SLOWFAST-ALPHA_4__SLOWFAST-BETA_INV_32__TRAIN-BATCH_SIZE_16__SOLVER-BASE_LR_0-001__SOLVER-OPTIMIZING_METHOD_adam

the accuracies are:
LGBP: 0.844
SF: 0.831
Fusion by MacroMajorityVoteCombiner: 0.909 (improvement of: 0.065)

After adding a second SF: SLOWFAST-ALPHA_4__SLOWFAST-BETA_INV_8__TRAIN-BATCH_SIZE_16__SOLVER-BASE_LR_0-001__SOLVER-OPTIMIZING_METHOD_sgd
with accuracy: 0.532

The accuracy of fusion by MacroMajorityVoteCombiner: 0.935 (improvement of 0.091)

adding one more SF improves the improvemnt by 0.026


"""