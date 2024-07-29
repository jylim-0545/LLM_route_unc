from datasets import load_dataset, Dataset, DownloadConfig
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llama_index.core.node_parser import SentenceSplitter

import faiss, json, os
import numpy as np
import random
from tqdm import tqdm


def load_tqa(num_q, is_random=False, q_number = None):
    
    docs = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    question = docs['question']
    answer = [ans['aliases'] + ans['normalized_aliases'] for ans in docs['answer']]
    
    if q_number:
        num_ls = []
        with open(q_number, "r") as fp:
            for line in fp:
                number = int(line.strip())
                num_ls.append(number-1)
        return  [question[ind] for ind in num_ls], [answer[ind] for ind in num_ls]
    
    else:
        if is_random:
            tmp = list(zip(question, answer))
            random.shuffle(tmp)
            question, answer = zip(*tmp)
        
        return question[:num_q], answer[:num_q]
  
  


def load_tqa_train(num_q, is_random=False, q_number = None):
    
    docs = load_dataset("mandarjoshi/trivia_qa", "rc", split='train')
    question = docs['question']
    answer = [ans['aliases'] + ans['normalized_aliases'] for ans in docs['answer']]
    
    if q_number:
        num_ls = []
        with open(q_number, "r") as fp:
            for line in fp:
                number = int(line.strip())
                num_ls.append(number-1)
        return  [question[ind] for ind in num_ls], [answer[ind] for ind in num_ls]
    
    else:
        if is_random:
            tmp = list(zip(question, answer))
            random.shuffle(tmp)
            question, answer = zip(*tmp)
        
        return question[:num_q], answer[:num_q]
  


  
def load_popqa(num_q, is_random=False, q_number = None):
    
    docs = load_dataset("akariasai/PopQA",  split='test')
    question = docs['question']
    answer = docs['possible_answers']
    
    if q_number:
        num_ls = []
        with open(q_number, "r") as fp:
            for line in fp:
                number = int(line.strip())
                num_ls.append(number-1)
        return  [question[ind] for ind in num_ls], [answer[ind] for ind in num_ls]
    
    else:
        if is_random:
            tmp = list(zip(question, answer))
            random.shuffle(tmp)
            question, answer = zip(*tmp)
        
        return question[:num_q], answer[:num_q]

def load_hqa(num_q, is_random=False, q_number = None):
    
    docs = load_dataset("hotpot_qa", "fullwiki", split='train')
    question = docs['question']
    answer = docs['answer']
    
    if q_number:
        num_ls = []
        with open(q_number, "r") as fp:
            for line in fp:
                number = int(line.strip())
                num_ls.append(number-1)
        return  [question[ind] for ind in num_ls], [[answer[ind]] for ind in num_ls]
    
    else:
        if is_random:
            tmp = list(zip(question, answer))
            random.shuffle(tmp)
            question, answer = zip(*tmp)
        
        return question[:num_q], [[ans] for ans in answer[:num_q]]
   
def load_nqa(num_q, is_random=False, q_number = None):
    
    print("NQA is loading")
    
    docs = load_dataset("nq_open",  split='train')
    question = docs['question']
    answer = docs['answer']
    
    if q_number:
        num_ls = []
        with open(q_number, "r") as fp:
            for line in fp:
                number = int(line.strip())
                num_ls.append(number-1)
        return  [question[ind] for ind in num_ls], [answer[ind] for ind in num_ls]

    else:
        if is_random:
            tmp = list(zip(question, answer))
            random.shuffle(tmp)
            question, answer = zip(*tmp)
        return question[:num_q], answer[:num_q]





if __name__ == '__main__':
    q, a = load_tqa(10)
    print(q[0])
    print(a[0])
    
    
    


