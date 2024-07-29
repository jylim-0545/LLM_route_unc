import time
import pandas as pd
import model
import dataset
import option
import util
import torch
import time
import numpy as np
from datetime import datetime
import math
from torch.nn.functional import softmax
from bart_score import BARTScorer
import sys
import ast
from tqdm import tqdm


args = sys.argv


df = pd.read_excel(f"{args[1]}.xlsx", index_col=0, dtype={'Answer': str})

gt_ls = df['GT'].apply(ast.literal_eval)
answer_ls = df['Answer']


bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart.pth')
    

rouge_ls = []
bart_ls =[]

for gt, ans in tqdm(zip(gt_ls, answer_ls), total = len(gt_ls)):
    ans = str(ans)
    rouge_ls.append(util.eval_rougu_L(ans, gt))
    bart_ls.append(util.eval_bartscore(ans, gt, bart_scorer))

df.insert(3, "Rouge-L", rouge_ls)
df.insert(4, "BART_score", bart_ls)

df.to_excel(f"{args[1]}.xlsx", index=True)   
