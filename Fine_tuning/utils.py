
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_metric
import os

label_list = ['correct', 'twin_2', 'twin_4', 'Sequence uncertainty', 'Perplexity', 'Max token uncertainty', 'Mean entropy', 'Max token entropy']

uncertainty_list = ['Sequence uncertainty', 'Perplexity', 'Max token uncertainty', 'Mean entropy', 'Max token entropy']

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
    
def draw_roc(correction, uncertainty_ls, name_ls, img_n, res_path):
    palette = sns.color_palette("hsv", len(uncertainty_ls))  # 'hsv' is a cyclic colormap suitable for categorical data

    plt.figure()
    
    for uncertainty, name, color in zip(uncertainty_ls, name_ls, palette):
        roc_auc, _, tpr, fpr, best_tpr, best_fpr = cal_roc(correction, uncertainty)
    
        
    
        plt.plot(fpr, tpr, color=color, lw=1, label=f'{name} (AUC = {roc_auc:.3f})')

    
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.savefig(f"test_result/{res_path}/{img_n}_ROC.png")

def cal_pr(correction, uncertainty):

    score = [-x for x in uncertainty]

    precision, recall, thresholds = precision_recall_curve(correction, score)
    pr_auc = auc(recall, precision)
    


    return pr_auc, precision, recall

def draw_pr(correction, uncertainty_ls, name_ls, img_n, res_path):
    palette = sns.color_palette("hsv", len(uncertainty_ls)) 
    plt.figure()
    
    for uncertainty, name, color in zip(uncertainty_ls, name_ls, palette):
        pr_auc, precision, recall = cal_pr(correction, uncertainty)
    
        plt.plot(recall, precision, color=color, lw=1, label=f'{name} (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.savefig(f"test_result/{res_path}/{img_n}_PR.png")


def split_path(path):
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            parts.append(tail)
        else:
            if path:
                parts.append(path)
            break
    parts.reverse()
    return parts



def preprocess_data(examples, tokenizer, label='correct', Hybrid = 0):
    tokenized_examples = tokenizer(examples['question'], truncation=True, padding='max_length')
    
    if label == 'correct':
        tokenized_examples['labels'] = [1.0 if correct else 0.0 for correct in examples['Correct']]
    elif label in ['Sequence uncertainty', 'Perplexity', 'Max token uncertainty', 'Mean entropy', 'Max token entropy']:
        # Metric 모드 처리
        if Hybrid == 0:
            tokenized_examples['labels'] = examples[label]
        else:
            correct = [1.0 if correct else (0.0 if Hybrid == 1 else -1.0) for correct in examples['Correct']]
            tokenized_examples['labels'] = [(1.0 - val) * cor for val, cor in zip(examples[label], correct)]
    elif label == 'twin_2' or label == 'twin_4':
        tokenized_examples['labels'] = examples['label_2'] if label == 'twin_2' else examples['label_4']
    else:
        raise ValueError(f"Invalid mode: {label}")
    
    return tokenized_examples




def compute_metrics(eval_pred):
    from sklearn.metrics import mean_squared_error
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    return {'mse': mse}

def compute_metrics_twin(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    predictions = torch.argmax(logits_tensor, dim=-1)
    metric = load_metric('accuracy')
    return metric.compute(predictions=predictions, references=labels)
