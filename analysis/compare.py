import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt



df_1 = pd.read_excel("opt_1_3b.xlsx")
df_2 = pd.read_excel("llama2_13b.xlsx")


Correct = df_1['Correct'].tolist()
Correction_1 = [1 if ind == True else 0 for ind in Correct]
seq_unc_1 = df_1['Sequence uncertainty'].tolist()

Correct = df_2['Correct'].tolist()
Correction_2 = [1 if ind == True else 0 for ind in Correct]
seq_unc_2 = df_2['Sequence uncertainty'].tolist()

cor_ls = []
unc_ls = []

cnt_draw_F = 0
cnt_draw_T = 0
cnt_win_1 = 0
cnt_win_2 = 0

unc_draw_F = []
unc_draw_T = []
unc_win_1 = []
unc_win_2 = []

for cor_1, cor_2, unc_1, unc_2 in zip(Correction_1, Correction_2, seq_unc_1, seq_unc_2):
    if cor_1 == 1 and cor_2 == 1: 
        cnt_draw_T += 1
        unc_draw_T.append(unc_1 - unc_2)
    elif cor_1 == 1 and cor_2 == 0: 
        cnt_win_1 += 1
        unc_win_1.append(unc_1-unc_2)
    elif cor_1 == 0 and cor_2 == 1: 
        cnt_win_2 += 1
        unc_win_2.append(unc_1 - unc_2)
    

    elif cor_1 == 0 and cor_2 == 0: cnt_draw_F += 1
    
    
    
    unc_ls.append(unc_1 - unc_2)
    

print(cnt_draw_T, cnt_draw_F, cnt_win_1, cnt_win_2)

