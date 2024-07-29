from transformers import GenerationConfig

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

def inverse_product_and_power(lst):
    N = len(lst)
    
    product = 1
    for num in lst:
        product *= 1 / num
    
    result = product ** (1 / N)
    
    return result

def question_and_test_input_bk(opt):
    
    LLM_pipeline, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    df = pd.DataFrame(columns = ['question', 'GT', "RAG_answer", "RAG_correct"])
    rag_ans_ls = []

    while (question := input("Type Question:")) != "exit":

        llm_p = util.generate_query_context(question, llm_prompt, False)
        rag_p = util.generate_query_context(question, rag_prompt, True, opt.top_k)


        for response in LLM_pipeline(llm_p):
            print(response)

        for response in LLM_pipeline(rag_p):
            print(response)





def question_and_test_input(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    df = pd.DataFrame(columns = ['question', 'GT', "RAG_answer", "RAG_correct"])
    rag_ans_ls = []

    while (question := input("Type Question:")) != "exit":

        llm_p = util.generate_query_context(question, llm_prompt, False)
        llm_p = tokenizer.encode(llm_p, return_tensors='pt').to("cuda")               
            
        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate = True,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id
        )


        max_new_tokens = 15

        input_len = len(llm_p[0])

        print("-" * 20 + "LLM" + "-" * 20)

        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens= max_new_tokens)

        tr_score = llm.compute_transition_scores(llm_response.sequences, llm_response.scores, normalize_logits=True).cpu()
        print(tokenizer.decode(llm_response.sequences[0, input_len:]))
        generated_tokens = llm_response.sequences[:, input_len:]
    
    

    
        for tok, score in zip(generated_tokens[0], tr_score[0]):
            print(f"| {tok:5d} | {tokenizer.decode(tok):10s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2f}")


        scores = llm_response.scores
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
        scores = scores.reshape(-1, scores.shape[-2], scores.shape[-1])
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.reshape(-1, scores.shape[-1]).cpu().numpy().transpose(1, 0)
        

        util.cal_score(tr_score[0].numpy(), scores)

        #print("Perplexity: ", torch.exp(tr_score.sum()*-1/len(tr_score[0])).item())



        print("-" * 20 + "RAG" + "-" * 20)


        rag_p = util.generate_query_context(question, rag_prompt, True, opt.top_k)
        rag_p = tokenizer.encode(rag_p, return_tensors='pt').to("cuda")               
        input_len = len(rag_p[0])

        rag_response = llm.generate(rag_p, generation_config = gen_config, max_new_tokens= max_new_tokens)

        print(tokenizer.decode(rag_response.sequences[0, input_len:]))


        
        

def question_and_test_input_cache(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    cache = None
    cuda_s = torch.cuda.Event(enable_timing=True)
    cuda_t = torch.cuda.Event(enable_timing=True)
    while (question := input("Type Question: ")) != "exit":

        #llm_p = tokenizer.encode(util.generate_query_context(question, llm_prompt, False), return_tensors='pt').to("cuda")
        #rag_p = tokenizer.encode(util.generate_query_context(question, rag_prompt, True, opt.top_k), return_tensors='pt').to("cuda")

        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=False,
            output_logits=False,
            output_hidden_states=False,
            return_dict_in_generate = True
        )
        
        prompt_in_chat_format_llm = [
        {
            "role": "system",
            "content": """
        Answer the question based on the given Context""",},
        {
            "role": "user",
            "content": """
            Context: {question}
            ---
            Question:""",
        },         
        ]


        if cache:
            #torch.cuda.empty_cache()
            llm_p = tokenizer.encode(question, return_tensors='pt').to("cuda")
            llm_p = llm_p[:, 1:]
            start =time.time()
            cuda_s.record()
            tmp = llm.generate(llm_p, generation_config = gen_config,  past_key_values = cache, max_new_tokens=1)
            end = time.time()
            cuda_t.record()
            torch.cuda.synchronize()
        else:
            #llm_p = tokenizer.apply_chat_template(prompt_in_chat_format_llm, tokenize = False)
            #llm_p = llm_p.format(question=question)
            llm_p = tokenizer.encode(question, return_tensors='pt').to("cuda")               
            
            #llm_p = llm_p[:, :-4]
            

            start =time.time()
            cuda_s.record()
            
            tmp = llm.generate(llm_p, generation_config = gen_config, max_new_tokens=20)      

            end = time.time()
            cuda_t.record()
            torch.cuda.synchronize()
            
        cache = tmp.past_key_values
        
        res_np = util.kv_to_np(cache)
        new_cache = util.np_to_kv(res_np)
        np.save("tmp.npy", res_np)


        print("Elapsed Time (ms): ", cuda_s.elapsed_time(cuda_t))
        print("Elapsed Time CPU: ", end - start)
        #print(cache[0][0].dtype)
        print("Cache shape: " + str(cache[0][1].shape))
        print("len of query token: " + str(llm_p.shape))
        print("len of output token: " + str(tmp.sequences[0].shape))
        #print(llm_p.shape)
        print(tokenizer.decode(tmp.sequences[0]))
        
        continue
        
        print(tmp.sequences)
        print(tmp.scores)
        print(tmp.logits)
        print(tmp.encoder_attentions)
        print(tmp.encoder_hidden_states)
        print(tmp.past_key_values)
        
        continue
        llm_response = tokenizer.decode(llm.generate(llm_p, generation_config = gen_config)[0], skip_special_tokens=True)
        rag_response = tokenizer.decode(llm.generate(rag_p, generation_config = gen_config)[0], skip_special_tokens=True)
        
        print(llm_response)
        print(rag_response)
        
def question_and_test_input_attention(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    while (question := input("Type Question: ")) != "exit":

        llm_p = tokenizer.encode(util.generate_query_context(question, llm_prompt, False), return_tensors='pt').to("cuda")
        rag_p = tokenizer.encode(util.generate_query_context(question, rag_prompt, True, opt.top_k), return_tensors='pt').to("cuda")

        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=False,
            output_logits=False,
            output_attentions = True,
            output_hidden_states= False,
            return_dict_in_generate = True
        )

        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens=20)      
        #rag_response = llm.generate(rag_p, generation_config = gen_config, max_new_tokens=100)      
        
        #print(tokenizer.decode(llm_response.sequences[0]))
        print(tokenizer.decode(llm_response.sequences[0]))
        str_ls = [tokenizer.decode([i]) for i in llm_response.sequences[0]]
        x = util.concat_attention(llm_response.attentions)
        util.draw_all_attention(x, str_ls)
        
        
        
        
        #print(len(tmp.attentions))
        #print(len(tmp.attentions[0]))
        
        continue
        
        for i in range(len(tmp.attentions)):
            print(tmp.attentions[i][5].shape)
        #print(len(tmp.sequences))
        print("len of query token: " + str(llm_p.shape))
        print("len of output token: " + str(tmp.sequences[0].shape))
        #print(llm_p.shape)
        print(tokenizer.decode(tmp.sequences[0]))
        str_ls = [tokenizer.decode([i]) for i in tmp.sequences[0]]
        
        x = util.concat_attention(tmp.attentions)
        util.draw_all_attention(x, str_ls)
        
        #util.draw_attention(x[0][0], "attention_tmp.jpg", str_ls)
        
        continue
        
        print(tmp.sequences)
        print(tmp.scores)
        print(tmp.logits)
        print(tmp.encoder_attentions)
        print(tmp.encoder_hidden_states)
        print(tmp.past_key_values)
        
        continue
        llm_response = tokenizer.decode(llm.generate(llm_p, generation_config = gen_config)[0], skip_special_tokens=True)
        rag_response = tokenizer.decode(llm.generate(rag_p, generation_config = gen_config)[0], skip_special_tokens=True)
        
        print(llm_response)
        print(rag_response)
        
def question_and_test_input_attention_tmp(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    while (question := input("Type Question: ")) != "exit":


        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=True,
            output_logits=False,
            output_attentions = True,
            output_hidden_states= False,
            return_dict_in_generate = True,
            pad_token_id = tokenizer.eos_token_id
        )

        llm_p = tokenizer.encode(question, return_tensors='pt').to("cuda")               
            
        max_new_tokens = 10

        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens= max_new_tokens)

    
        tr_score = llm.compute_transition_scores(llm_response.sequences, llm_response.scores, normalize_logits=True).cpu()
        
        print(torch.exp(tr_score.sum()*-1/len(tr_score[0])))
        
        print(tr_score)
        
        continue
        
        generated_tokens = llm_response.sequences[0, -1*max_new_tokens:]
        generated_string = [tokenizer.decode(tok) for tok in generated_tokens]
        
        
        
        score_ls = np.exp(tr_score[0])
        
        perplexity = inverse_product_and_power(list(score_ls))
        
        for tok, str, score in zip(generated_tokens, generated_string, score_ls):
            print(f"| {tok:5d} | {str:10s} | {score:.3f}")
        
    
        
        print(perplexity)
        
        
        

        '''for at in llm_response.attentions:
            print(at[0].shape)
        
        
        
        str_ls = [tokenizer.decode([i]) for i in llm_response.sequences[0]]
        
        str_ls = str_ls[:-1]
        
        M = 3
        
        new_str_ls = []
        N = len(str_ls) - 1
        
        for i in range(N//M+1):
            new_str_ls.append(' '.join(str_ls[i*M:min((i+1)*M, N)]))
            
        
        x = util.concat_attention_no_head(llm_response.attentions)
        
        
        
        util.draw_all_attention_no_head(x, str_ls, M)'''
        
def question_and_test_input_perplexity(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    while (question := input("Type Question: ")) != "exit":

        llm_p = tokenizer(question, return_tensors='pt')             
            
        max_length = llm.config.max_position_embeddings
        stride = 512

        nlls = []
        for i in tqdm(range(0, llm_p.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, llm_p.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = llm_p.input_ids[:, begin_loc:end_loc].to("cuda")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = llm(input_ids, labels=target_ids)

                neg_log_likelihood = outputs[0] * trg_len

            print(len(outputs))
            print(outputs.loss)
            print(outputs.logits)
            print(outputs.logits.shape)
            
            

            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc).cpu().int()
        print("PPL: " + str(ppl))

  



def question_and_test_context_score(opt):
    


    if opt.ds == "tqa":
        question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "popqa":
        question_ls, gt_ls = dataset.load_popqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "hqa":
        question_ls, gt_ls = dataset.load_hqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "nqa":
        question_ls, gt_ls = dataset.load_nqa(opt.num_q, opt.random, opt.q_number)        

    df = pd.DataFrame(columns = ['Question', 'GT', 'Context','Score'])
    
    score_ls = []
    context_ls = []

    for q in tqdm(question_ls):
        contexts, scores = util.search_context(q, opt.top_k)
        score_ls.append(scores[0])
        context_ls.append(contexts[0])
    
    for q, gt, context, score in zip(question_ls, gt_ls, context_ls, score_ls):
        df.loc[len(df)] = [q, gt, context, score]
    
    df.to_excel(f"log/{opt.ds}.xlsx", index=True)   
    
def question_and_test_ds(opt):
    
    LLM_pipeline, tokenizer, _, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    prompt = rag_prompt if opt.is_rag else llm_prompt
    
    if opt.ds == "tqa":
        question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "popqa":
        question_ls, gt_ls = dataset.load_popqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "hqa":
        question_ls, gt_ls = dataset.load_hqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "nqa":
        question_ls, gt_ls = dataset.load_nqa(opt.num_q, opt.random, opt.q_number)        

    rag_ans_ls = []
    df = pd.DataFrame(columns = ['question',"Answer", "Correct"])
    
    for response in tqdm(LLM_pipeline(util.generate_query_context_list(question_ls, prompt, opt.is_rag, opt.top_k)), total = opt.num_q):
        rag_ans_ls.append(response[0]['generated_text'])
    
    rag_correct_ls = [util.eval_EM(ans, gt) for ans, gt in zip(rag_ans_ls, gt_ls)]
    for q, rag_ans, rag_correct in zip(question_ls, rag_ans_ls,  rag_correct_ls, ):
        df.loc[len(df)] = [q, rag_ans,  rag_correct]

    df.to_excel(f"{opt.log_dir}/{opt.ds}.xlsx", index=True)   



def question_and_test_ds_perplexity(opt):
    
    _, tokenizer, llm, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    prompt = rag_prompt if opt.is_rag else llm_prompt
    
    if opt.ds == "tqa":
        question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "popqa":
        question_ls, gt_ls = dataset.load_popqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "hqa":
        question_ls, gt_ls = dataset.load_hqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "nqa":
        question_ls, gt_ls = dataset.load_nqa(opt.num_q, opt.random, opt.q_number)        

    rag_ans_ls = []
    score_lss = []
    str_ls = []
    first_prob_ls = []
    df = pd.DataFrame(columns = ['question',"Answer", "Correct", "Prob_first", "Score", "String"])
    
    for question in tqdm(question_ls):

        llm_p = util.generate_query_context(question, prompt, opt.is_rag, opt.top_k)
        llm_p = tokenizer.encode(llm_p, return_tensors='pt').to("cuda")               

        gen_config = GenerationConfig(
            do_sample = False,
            output_scores=True,
            output_logits=False,
            output_attentions = True,
            output_hidden_states= False,
            return_dict_in_generate = True,
            pad_token_id = tokenizer.eos_token_id
        )



        max_new_tokens = 100

        input_len = len(llm_p[0])

        llm_response = llm.generate(llm_p, generation_config = gen_config, max_new_tokens= max_new_tokens)

        ans = tokenizer.decode(llm_response.sequences[0, input_len:])
        rag_ans_ls.append(ans)

        tr_score = llm.compute_transition_scores(llm_response.sequences, llm_response.scores, normalize_logits=True).cpu()        
        generated_tokens = llm_response.sequences[0, input_len:]
        generated_string = [tokenizer.decode(tok) for tok in generated_tokens]
        score_ls = np.exp(tr_score[0])
        score_lss.append(score_ls.tolist())
        str_ls.append(generated_string)
        first_prob_ls.append(score_ls[0].item() if opt.llm_n == 'llama3_8b' else score_ls[1].item())
        
    rag_correct_ls = [util.eval_EM(ans, gt) for ans, gt in zip(rag_ans_ls, gt_ls)]
    for q, rag_ans, rag_correct, f_b, sc, strr in zip(question_ls, rag_ans_ls,  rag_correct_ls,  first_prob_ls, score_lss, str_ls):
        df.loc[len(df)] = [q, rag_ans,  rag_correct,f_b, sc, strr]


    df.to_excel(f"{opt.log_dir}/{opt.ds}.xlsx", index=True)   





def question_and_test_ds_average(opt):
    
    LLM_pipeline, tokenizer, _, llm_prompt, rag_prompt = model.load_model(opt.llm_n)
    
    prompt = rag_prompt if opt.is_rag else llm_prompt
    
    if opt.ds == "tqa":
        question_ls, gt_ls = dataset.load_tqa(opt.num_q, opt.random, opt.q_number)
    elif opt.ds == "popqa":
        question_ls, gt_ls = dataset.load_popqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "hqa":
        question_ls, gt_ls = dataset.load_hqa(opt.num_q, opt.random, opt.q_number)        
    elif opt.ds == "nqa":
        question_ls, gt_ls = dataset.load_nqa(opt.num_q, opt.random, opt.q_number)        

    df = pd.DataFrame(columns = ['question', "RAG_correct"])
    
    num_repeat = 10
    
    all_rag_correct_ls = np.zeros((opt.num_q))
    
    for _ in range(num_repeat):
    
        rag_ans_ls = []
        
        for response in tqdm(LLM_pipeline(util.generate_query_context_list(question_ls, prompt, opt.is_rag, opt.top_k)), total = opt.num_q):
            rag_ans_ls.append(response[0]['generated_text'])
        rag_correct_ls = [1 if util.eval_EM(ans, gt) else 0 for ans, gt in zip(rag_ans_ls, gt_ls)]
        all_rag_correct_ls += np.array(rag_correct_ls)
    
    all_rag_correct_ls /= num_repeat
    
    for q, rag_correct in zip(question_ls, all_rag_correct_ls):
        df.loc[len(df)] = [q, rag_correct]

    df.to_excel(f"{opt.log_dir}/{opt.ds}.xlsx", index=True)   



if __name__ == '__main__':
    
    opt = option.parse_argument()
    
    
    
    current_time_str = datetime.now().strftime('%m_%d_%H_%M')
    log_dir = f"log/{current_time_str}"
    os.mkdir(log_dir)
    util.opt_log(opt, log_dir)

    opt.log_dir = log_dir

    if opt.use_input:
        question_and_test_input(opt)         
    else:
        question_and_test_ds_perplexity(opt)    



    
    
    
    

