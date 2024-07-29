from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification

import os

from tqdm import tqdm

import time
import pandas as pd

import model
import dataset
from lib_retriever import Retriever
import option
     
import util



def question_and_test_input(opt):
    
    LLM_pipeline, tokenizer = model.load_model(opt.llm_n)
    
    retriever = Retriever(opt.embed_n, opt.dataset_n, opt.chunk_len,  opt.index_n, tokenizer, opt.llm_n, opt.rerank)

    while (question := input("Type Question:")) != "exit":

        llm_prompt = retriever.prompt_llm.format(question = question)
        rag_prompt = retriever.retrieve_and_prompt(opt.top_k,question)
        
        llm_response = LLM_pipeline(llm_prompt)[0]['generated_text']
        rag_response = LLM_pipeline(rag_prompt)[0]['generated_text']
        
        log_n = f"log/log_{opt.dataset_n}_{opt.embed_n}_{opt.index_n}_{opt.llm_n}_input" + ("_rerank.xlsx" if opt.rerank else ".xlsx")
        if os.path.exists(log_n):
            df = pd.read_excel(log_n)
            df.drop(['Unnamed: 0'], axis = 1, inplace = True)
        else:
            df = pd.DataFrame(columns = ['question', 'Context', 'LLM_answer', "RAG_answer"])
            


        df.loc[len(df)] = [question,  retriever.context_ls, llm_response, rag_response]

        retriever.context_ls = []

        df.to_excel(log_n, index=True)   

def question_and_test_tqa_RAG(opt):
    
    LLM_pipeline, tokenizer = model.load_model(opt.llm_n)
    
    retriever = Retriever(opt.embed_n, opt.dataset_n, opt.chunk_len,  opt.index_n, tokenizer, opt.llm_n, opt.rerank)
       
       
    #logging
    log_n = f"log/log_{opt.dataset_n}_{opt.chunk_len}_{opt.embed_n}_{opt.index_n}_{opt.llm_n}_TQA" + ("_rerank.xlsx" if opt.rerank else ".xlsx")
    if os.path.exists(log_n):
        df = pd.read_excel(log_n)
        df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    else:
        df = pd.DataFrame(columns = ['question', 'GT', 'Context', "RAG_answer", "RAG_correct", "Scores", "T_retrieve", "T_rag"])
      
    question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random)

    rag_ans_ls = []
    retrieve_time_ls = []
    rag_time_ls = []
    score_ls = []

    if opt.rerank:
        start_time = time.time()
        for response in tqdm(LLM_pipeline(retriever.rag_rerank_prompt_seq(question_ls, opt.top_k)), total = opt.num_q):
            end_time = time.time()
            rag_time_ls.append(end_time - start_time)
            retrieve_time_ls.append(retriever.retrieve_time)
            retrieve_time_ls.append(retriever.retrieve_time/1000)
            rag_ans_ls.append(response[0]['generated_text'])
            score_ls.append(retriever.scores)
            start_time = time.time()
    else:
        start_time = time.time()
        for response in tqdm(LLM_pipeline(retriever.rag_prompt_seq(question_ls, opt.top_k)), total = opt.num_q):
            end_time = time.time()
            rag_time_ls.append(end_time - start_time)
            retrieve_time_ls.append(retriever.retrieve_time/1000)
            rag_ans_ls.append(response[0]['generated_text'])
            score_ls.append(retriever.scores)
            start_time = time.time()
    

    rag_correct_ls = [util.eval_EM(ans, gt) for ans, gt in zip(rag_ans_ls, gt_ls)]
    
    context_ls = retriever.context_ls
    
    for q, gt, ct,  rag_ans, rag_correct, scores, retrieve_time, rag_time in zip(question_ls, gt_ls, context_ls, rag_ans_ls,  rag_correct_ls, score_ls, retrieve_time_ls, rag_time_ls):
        df.loc[len(df)] = [q, gt, ct, rag_ans,  rag_correct, scores, retrieve_time, rag_time]

    df.to_excel(log_n, index=True)   

def question_and_test_tqa_LLM(opt):
    
    LLM_pipeline, tokenizer = model.load_model(opt.llm_n)
    
    retriever = Retriever(opt.embed_n, opt.dataset_n, opt.chunk_len,  opt.index_n, tokenizer, opt.llm_n, opt.rerank, opt.no_rag)
       
    #logging
    log_n = f"log/log_{opt.llm_n}_TQA.xlsx"
    
    if os.path.exists(log_n):
        df = pd.read_excel(log_n)
        df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    else:
        df = pd.DataFrame(columns = ['question', 'GT', 'LLM_answer', "LLM_correct", "T_LLM"])
      
    question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random)

    llm_ans_ls = []
    llm_time_ls = []

    start_time = time.time()
    for response in tqdm(LLM_pipeline(retriever.llm_prompt_seq(question_ls)), total = opt.num_q):
        end_time = time.time()
        llm_time_ls.append(end_time - start_time)
        llm_ans_ls.append(response[0]['generated_text'])
        start_time = time.time()    
    

    llm_correct_ls = [util.eval_EM(ans, gt) for ans, gt in zip(llm_ans_ls, gt_ls)]


    for q, gt,  llm_ans, llm_correct, llm_t in zip(question_ls, gt_ls, llm_ans_ls, llm_correct_ls, llm_time_ls):
        df.loc[len(df)] = [q, gt, llm_ans, llm_correct, llm_t]

    df.to_excel(log_n, index=True)   



if __name__ == '__main__':
    
    opt = option.parse_argument()
    

    if opt.use_input:
        question_and_test_input(opt)         
    elif opt.no_rag:
        question_and_test_tqa_LLM(opt) 
    else:
        question_and_test_tqa_RAG(opt)    



    
    
    
    

