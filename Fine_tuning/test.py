import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import sys
import os
import utils
import numpy as np
import utils
import option




def log_metric(question, pred, lab, correction_GT, res_path):
    df = pd.DataFrame(columns = ['Question', 'Prediction', 'Label', 'Correction'])
    for q, p, l, gt in zip(question, pred, lab, correction_GT):
        df.loc[len(df)] = [q, p, l, gt]
    os.makedirs(f"test_result/{res_path}/log/", exist_ok = True)
    df.to_excel(f"test_result/{res_path}/log/{label}_{val_file_n}.xlsx")

def log_metric_twin(question, pred, lab,res_path):
    df = pd.DataFrame(columns = ['Question', 'val', 'Prediction', 'Label', 'Correction'])
    for q, p, l in zip(question, pred, lab):
        pr = np.argmax(np.array(p))
        df.loc[len(df)] = [q, p, pr, l, pr == l]
    os.makedirs(f"test_result/{res_path}/log/", exist_ok = True)
    df.to_excel(f"test_result/{res_path}/log/{label}_{val_file_n}.xlsx")


def cal_correct(pred, label):
    TP, FP, FN, TN = 0, 0, 0, 0
    for p, l in zip(pred, label):
        p = 1 if p > 0.5 else 0
        
        if p == 1 and l == 1: TP+=1
        elif p== 1 and l == 0: FP+=1
        elif p == 0 and l == 1: FN +=1
        else: TN +=1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR
        
def log_correct(question, pred, label, res_path):
    df = pd.DataFrame(columns = ['Question', 'P_origin', 'Prediction', 'Label', 'diff'])

    for q, p, l in zip(question, pred, label):
        p_hard = 1 if p > 0.5 else 0
        
        df.loc[len(df)] = [q, p, p_hard , l, p_hard == l]

    TPR, FPR = cal_correct(pred, label)
    
    df.loc[len(df)] = ["0", "TPR", TPR, "FPR", FPR]

    os.makedirs(f"test_result/{res_path}/log/", exist_ok = True)

    df.to_excel(f"test_result/{res_path}/log/{val_file_n}.xlsx")

def eval(label, ds, model, tokenizer):
    
    tokenized_dataset = ds.map(lambda x: utils.preprocess_data(x, tokenizer, label, opt.hybrid), batched=True)

    training_args = TrainingArguments(
        output_dir='./trash',          # output directory
        evaluation_strategy='epoch',     # evaluation strategy to use
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,   # batch size for evaluation
        num_train_epochs=3,              # number of training epochs
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics= utils.compute_metrics_twin if label == 'twin_2' or label == 'twin_4' else utils.compute_metrics
    )

    eval_results = trainer.predict(tokenized_dataset)
    
    return eval_results

opt = option.parse_argument()
path_ls = utils.split_path(opt.model_path)

label = opt.label
df = pd.read_excel(f'data/{opt.file_n}.xlsx')

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(opt.model_path)

# Load the DeBERTa model for regression (1 output unit)
model = AutoModelForSequenceClassification.from_pretrained(
    opt.model_path, 
    num_labels=4 if label == 'twin_4' else 2 if label == 'twin_2' else 1
)

eval_results = eval(label, dataset, model, tokenizer)


if not (label in ['twin_2', 'twin_4', 'relative uncertainty', 'relative uncertainty 1', 'relative uncertainty 2']):
    Correct = df['Correct'].tolist()
    Correction = [1 if ind == True else 0 for ind in Correct]
if label in ['relative uncertainty', 'relative uncertainty 1', 'relative uncertainty 2']:
    Correction = df['label_2'].tolist()


if label == 'correct':
    res_dir = f"{path_ls[1]}/Correct/{path_ls[3]}"
    #log_correct(df['question'], eval_results[0], eval_results[1], res_dir)
    log_metric(df['question'], eval_results[0], eval_results[1], df['Correct'], res_dir)
    utils.draw_roc(Correction, [-1*eval_results[0]], ['Prediction'], label, res_dir)
    utils.draw_pr(Correction, [-1*eval_results[0]], ['Prediction'], label, res_dir)
elif label == 'Seq_weight_0':
    res_dir = f"{path_ls[1]}/{path_ls[3]}"
    log_metric(df['question'], eval_results[0], eval_results[1], df['Correct'], res_dir)
    utils.draw_roc(Correction, [-1*eval_results[0]], ['Prediction'], label, res_dir)
    utils.draw_pr(Correction, [-1*eval_results[0]], ['Prediction'], label, res_dir)
    
elif label == 'Seq_weight_1':
    res_dir = f"{path_ls[1]}/{path_ls[3]}"
    Correction = [1 if ind == True else -1 for ind in Correct]
    log_metric(df['question'], eval_results[0], eval_results[1], df['Correct'], res_dir)
    utils.draw_roc(Correction, [-1*eval_results[0]], ['Prediction'], label, res_dir)
elif label == 'twin_2' or label == 'twin_4':
    res_dir = f"{path_ls[1]}/{path_ls[3]}"
    log_metric_twin(df['question'], eval_results[0], eval_results[1],  res_dir)
else:
    res_dir = f"{path_ls[1]}/{path_ls[3]}"
    log_metric(df['question'], eval_results[0], eval_results[1], Correction, res_dir)
    utils.draw_roc(Correction, [eval_results[0], eval_results[1]], ['Prediction', 'GT'], label, res_dir)
    utils.draw_pr(Correction, [eval_results[0], eval_results[1]], ['Prediction', 'GT'], label, res_dir)
    
    


