import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import sys
import utils
import option

opt = option.parse_argument()

label = opt.label
model_n = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_n)

# Load your dataset
df = pd.read_excel(f'data/{opt.file_n}.xlsx')
ds = Dataset.from_pandas(df)

if opt.file_2_n != None:
    df_2 = pd.read_excel(f'data/{opt.file_2_n}.xlsx')
    ds_2 = Dataset.from_pandas(df_2)


tokenized_dataset = ds.map(lambda x: utils.preprocess_data(x, tokenizer, label, opt.hybrid), batched=True)



# Split the dataset
train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.01).values()


# Load the DeBERTa model for regression (1 output unit)

model = AutoModelForSequenceClassification.from_pretrained(
    opt.model_path, 
    num_labels=4 if label == 'twin_4' else 2 if label == 'twin_2' else 1
)


training_args = TrainingArguments(
    output_dir=f'./model/{opt.file_n}/{label}',          # output directory
    evaluation_strategy='epoch',     # evaluation strategy to use
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    num_train_epochs=6,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    learning_rate= 2e-5,
    lr_scheduler_type='constant',
    logging_dir='./logs',            # directory for storing logs
    logging_steps=2000,
    save_steps=12000,
    #save_steps = 300
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics= utils.compute_metrics_twin if label == 'twin_2' or label == 'twin_4' else utils.compute_metrics)

trainer.train()

