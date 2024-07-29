import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_excel('relative uncertainty_opt_13b_llama2_7b_train.xlsx')
pred = df['Prediction']

df_2 = pd.read_excel('relative uncertainty 1_opt_13b_llama2_7b_train.xlsx')
pred_2 = df_2['Prediction']


df_3 = pd.read_excel('relative uncertainty 2_opt_13b_llama2_7b_train.xlsx')

pred_3 = df_3['Prediction']

Correction = df['Correction']




def cal_roc(correction, uncertainty):

    score = [-x for x in uncertainty]

    fpr, tpr, thresholds = roc_curve(correction, score)
    roc_auc = auc(fpr, tpr)
    
    J = tpr - fpr
    best_threshold_index = np.argmax(J)
    best_threshold = thresholds[best_threshold_index]
    best_tpr = tpr[best_threshold_index]
    best_fpr = fpr[best_threshold_index]
    
    return roc_auc, best_threshold, tpr, fpr, best_tpr, best_fpr
    
def draw_roc(correction, uncertainty_ls, name_ls, img_n):
    palette = sns.color_palette("hsv", len(uncertainty_ls))  # 'hsv' is a cyclic colormap suitable for categorical data

    plt.figure()
    
    for uncertainty, name, color in zip(uncertainty_ls, name_ls, palette):
        roc_auc, _, tpr, fpr, best_tpr, best_fpr = cal_roc(correction, uncertainty)
    
        plt.plot(fpr, tpr, color=color, lw=1, label=f'{name} (AUC= {roc_auc:.3f})')

    
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.savefig(f"{img_n}.png")




draw_roc(Correction, [pred, pred_2, pred_3], ['Relative uncertainty predictor', 'Relative Score 1 predictor', 'Relative Score 2 predictor'], "Llama3_8B")


