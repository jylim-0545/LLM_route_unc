import pandas as pd
import sys
import os
import utils



args = sys.argv

model_n = args[1]

val_file_n = args[2]

metric = ['correct', 'Sequence uncertainty', 'Perplexity', 'Max token uncertainty', 'Mean entropy', 'Max token entropy', 'Seq_weight_0', 'Seq_weight_1']

label = metric[int(args[3])]



root_path = f"test_result/{model_n}"

if label == 'correct':
    root_path += "/Correct"


dir_ls = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d != "Correct"]

unc_ls = []

for dirs in dir_ls:
    df = pd.read_excel(f"{root_path}/{dirs}/log/{label}_{val_file_n}.xlsx")
    pred = df['Prediction']
    if label in ['correct', 'Seq_weight_0', 'Seq_weight_1']:
        pred = -1 * pred
    unc_ls.append(pred)

if label not in ['correct', 'Seq_weight_0', 'Seq_weight_1']:
    GT_unc = df['Label']
    unc_ls.append(GT_unc)
    dir_ls.append("GT")

Correct = df['Correction'].tolist()
Correction = [1 if ind == True else 0 for ind in Correct]

utils.draw_roc(Correction, unc_ls, dir_ls, label, f"{model_n}" if label != 'correct' else f"{model_n}/Correct")
