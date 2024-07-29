from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

import os
from tqdm import tqdm

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

def inverse_product_and_power(lst):
    N = len(lst)
    
    product = 1
    for num in lst:
        product *= 1 / num
    
    result = product ** (1 / N)
    
    return result

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        
        for seq in input_ids:
            for stop in self.stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    return True
        return False

llm_p_bk = "Question: At what temperature does water boil? Answer: 100°C \nQuestion: When did World War II start and end? Answer: 1939-1945 \nQuestion: Who wrote '1984'? Answer: George Orwell \nQuestion: What is the basic principle of artificial intelligence Answer: Machine learning \nQuestion: Which symphony is Beethoven's 'Fate' symphony? Answer: Fifth \nQuestion: What is the highest mountain in the world? Answer: Everest \n Question: What does GDP stand for? Answer: Gross Domestic Product \nQuestion: Who painted the 'Mona Lisa'? Answer: Leonardo da Vinci \nQuestion: What is the capital of France? Answer: Paris \nQuestion: What is the chemical symbol for gold? Answer: Au \n"
    
llm_p_bk_2 = "Question: At what temperature does water boil? Answer: 100°C Question: When did World War II start and end? Answer: 1939-1945 Question: Who wrote '1984'? Answer: George Orwell Question: What is the basic principle of artificial intelligence Answer: Machine learning Question: Which symphony is Beethoven's 'Fate' symphony? Answer: Fifth Question: What is the highest mountain in the world? Answer: Everest Question: What does GDP stand for? Answer: Gross Domestic Product Question: Who painted the 'Mona Lisa'? Answer: Leonardo da Vinci "

def clear_generation_ans(seq, scores, entropy_score, input_len, ans_chk):
    for ind in range(input_len, len(seq), 1):
        for ans_ids in ans_chk:
            if seq[ind: ind + len(ans_ids)].tolist() == ans_ids:
                seq_tok = seq[ind + len(ans_ids):]
                scores = scores[len(ans_ids):]
                entropy_score_1 = entropy_score[len(ans_ids):]
                return seq_tok, scores, entropy_score_1
    return seq[input_len+2:], scores[2:], entropy_score[2:]

def clear_generation_ans_end(seq, scores, entropy_score, ans_chk):
    for ind in range(len(seq)):
        for ans_ids in ans_chk:
            if seq[ ind : ind + len(ans_ids) ].tolist() == ans_ids:
                seq_tok = seq[ :  ind]
                scores_1 = scores[: ind]
                entropy_score_1 = entropy_score[: ind]
                return seq_tok, scores_1, entropy_score_1
    return seq, scores, entropy_score    

def clear_generation_stop(seq_tok, scores, entropy_score, stop_chk):
    for std_ids in stop_chk:
        if seq_tok[-len(std_ids):].tolist() == std_ids:
            seq_tok = seq_tok[:-len(std_ids)]
            scores = scores[:-len(std_ids)]
            entropy_score = entropy_score[:-len(std_ids)]
            return seq_tok, scores, entropy_score
    return seq_tok, scores, entropy_score

def clear_generation(seq, scores, entropy_scores, input_len, ans_chk, stop_chk):
    
    seq_tok, score, entropy_score = clear_generation_ans(seq, scores, entropy_scores, input_len, ans_chk)

    #seq_tok, score, entropy_score = clear_generation_ans_end(seq_tok, score, entropy_score, ans_chk)

    return clear_generation_stop(seq_tok, score, entropy_score, stop_chk)


def question_and_test_input(opt):
    
    _, tokenizer, llm, _, _ = model.load_model(opt.llm_n)
    
    num_candidates = 1



    while (question := input("Type Question:")) != "exit":
    
        llm_p =  "Question: At what temperature does water boil? Answer: 100°C \nQuestion: When did World War II start and end? Answer: 1939-1945 \nQuestion: Who wrote '1984'? Answer: George Orwell \nQuestion: What is the basic principle of artificial intelligence Answer: Machine learning \nQuestion: Which symphony is Beethoven's 'Fate' symphony? Answer: Fifth \nQuestion: What is the highest mountain in the world? Answer: Everest \n Question: What does GDP stand for? Answer: Gross Domestic Product \nQuestion: Who painted the 'Mona Lisa'? Answer: Leonardo da Vinci \nQuestion: What is the capital of France? Answer: Paris \nQuestion: What is the chemical symbol for gold? Answer: Au \n"
        
        llm_p = llm_p +  "Question: " +question
        
        llm_p = tokenizer.encode(llm_p, return_tensors='pt').to("cuda")               
        
        stop_words = [" \n", "\n", "\n "]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]

        stop_words_ids = [ids.unsqueeze(0) if ids.dim() == 0 else ids for ids in stop_words_ids]
        stop_words_ids += stop_words_ids[0][-1:].unsqueeze(0)
        stop_words_ids_chk = [tensor.tolist() for tensor in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


        answer_words = [" Answer:", "Answer:", "Answer: "]
        answer_words_ids = [tokenizer(answer_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for answer_word in answer_words]
        answer_words_ids = [ids.unsqueeze(0) if ids.dim() == 0 else ids for ids in answer_words_ids]
        answer_words_ids_chk = [tensor.tolist() for tensor in answer_words_ids]

        #print(answer_words_ids_chk)
        
        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=True,
            output_logits=True,
            num_return_sequences = num_candidates,
            return_dict_in_generate = True)

        max_new_tokens = 20

        input_len = len(llm_p[0])

        print("-" * 50 + "LLM" + "-" * 50)

        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id, stopping_criteria = stopping_criteria)

        tr_score = llm.compute_transition_scores(llm_response.sequences, llm_response.scores, normalize_logits=True).cpu().numpy()
        scores = llm_response.scores
        scores = torch.stack(scores).transpose(0, 1)
        entropy_scores = torch.nn.functional.softmax(scores, dim=2).cpu().numpy()
        
        for seq, score_1, entropy_score_1 in zip(llm_response.sequences, tr_score, entropy_scores):
            
            seq_tok, score_f, entropy_score_f = clear_generation(seq, score_1, entropy_score_1, input_len, answer_words_ids_chk, stop_words_ids_chk)

            #for tok, score in zip(seq_tok, score_f):
            #    print(f"tok id: {tok} | word: {tokenizer.decode(tok)} | score: {score:.2f} | prob: {np.exp(score):.4f}")
            #print(tokenizer.decode(seq))

            print(tokenizer.decode(seq_tok))

            t, _, _, _, _ = util.cal_score(score_f, entropy_score_f)
            print(t)
        
            print("#" * 100)
            

def question_and_test_ds(opt):
    
    _, tokenizer, llm, _, _ = model.load_model(opt.llm_n)
    
    if opt.ds == "tqa":
        question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "tqa_train":
        question_ls, gt_ls = dataset.load_tqa_train(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "popqa":
        question_ls, gt_ls = dataset.load_popqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "hqa":
        question_ls, gt_ls = dataset.load_hqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "nqa":
        question_ls, gt_ls = dataset.load_nqa(opt.num_q, opt.random, opt.q_number)        

    df = pd.DataFrame(columns = ['question', "GT", "Answer", "Correct", "Rouge-L", "BART_score", "Sequence uncertainty", "Perplexity", "Max token uncertainty", "Mean entropy", "Max token entropy"])
    
    llm_p_base = "Question: At what temperature does water boil? Answer: 100°C \nQuestion: When did World War II start and end? Answer: 1939-1945 \nQuestion: Who wrote '1984'? Answer: George Orwell \nQuestion: What is the basic principle of artificial intelligence Answer: Machine learning \nQuestion: Which symphony is Beethoven's 'Fate' symphony? Answer: Fifth \nQuestion: What is the highest mountain in the world? Answer: Everest \n Question: What does GDP stand for? Answer: Gross Domestic Product \nQuestion: Who painted the 'Mona Lisa'? Answer: Leonardo da Vinci \nQuestion: What is the capital of France? Answer: Paris \nQuestion: What is the chemical symbol for gold? Answer: Au \n"
        
    stop_words = [" \n", "\n", "\n "]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stop_words_ids = [ids.unsqueeze(0) if ids.dim() == 0 else ids for ids in stop_words_ids]
    stop_words_ids += stop_words_ids[0][-1:].unsqueeze(0)

    stop_words_ids_chk = [tensor.tolist() for tensor in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    answer_words = [" Answer:", "Answer:", "Answer: "]
    answer_words_ids = [tokenizer(answer_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for answer_word in answer_words]
    answer_words_ids = [ids.unsqueeze(0) if ids.dim() == 0 else ids for ids in answer_words_ids]
    answer_words_ids_chk = [tensor.tolist() for tensor in answer_words_ids]


    gen_config = GenerationConfig(
        do_sample = False,
        output_scores=True,
        output_logits=True,
        num_return_sequences = 1,
        return_dict_in_generate = True
    )
    
    max_new_tokens = 50
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart.pth')
    
    for question, gt in tqdm(zip(question_ls, gt_ls), total = len(question_ls)):
        llm_p = llm_p_base + "Question: " + question
    
        llm_p = tokenizer.encode(llm_p, return_tensors='pt').to("cuda")               
        input_len = len(llm_p[0])
    
        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id, stopping_criteria = stopping_criteria)
    
        tr_score = llm.compute_transition_scores(llm_response.sequences, llm_response.scores, normalize_logits=True).cpu().numpy()
        scores = llm_response.scores
        scores = torch.stack(scores).transpose(0, 1)
        entropy_scores = torch.nn.functional.softmax(scores, dim=2).cpu().numpy()
        
        for seq, score_1, entropy_score_1 in zip(llm_response.sequences, tr_score, entropy_scores):
            
            seq_tok, score_f, entropy_score_f = clear_generation(seq, score_1, entropy_score_1, input_len, answer_words_ids_chk, stop_words_ids_chk)


            a, b, c, d, e = util.cal_score(score_f, entropy_score_f)
        
            ans = tokenizer.decode(seq_tok)

            if len(seq_tok) != 0:
                break

        df.loc[len(df)] = [question, gt, ans, util.eval_EM(ans, gt), util.eval_rougu_L(ans, gt), util.eval_bartscore(ans, gt, bart_scorer), a, b, c, d, e]

    df.to_excel(f"{opt.log_dir}/{opt.ds}.xlsx", index=True)   


def question_and_test_input_bart(opt):

    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart.pth')
    
    
    while (text_1 := input("Type Text 1:")) != "exit":
        text_2  = input("Type Text 2:")
        
        res = bart_scorer.score([text_1], [text_2])[0]
        
        print(res[0])
        






if __name__ == '__main__':
    
    opt = option.parse_argument()
    
    
    


    if opt.use_input:
        question_and_test_input_bart(opt)         
    else:
        current_time_str = datetime.now().strftime('%m_%d_%H_%M')
        log_dir = f"log/{current_time_str}_{opt.llm_n}"
        os.mkdir(log_dir)
        util.opt_log(opt, log_dir)

        opt.log_dir = log_dir
        question_and_test_ds(opt)    



    
    
    
    

