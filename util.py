
import sys
sys.path.append("db_server")
from db_server import faiss_pb2_grpc, faiss_pb2
import grpc
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import yaml
from transformers import StoppingCriteria, AutoTokenizer
from typing import List
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.nn.functional import softmax

def prompt_template(model_n, tokenizer):


    prompt_in_chat_format_rag = [
    {
        "role": "system",
        "content": """Answer the given question with a single word or phrase""",
        },
        {
            "role": "user",
            "content": """Context: {context}
    Question: {question}""",
        },         
    ]
    
    prompt_in_chat_format_llm = [
        {
            "role": "system",
            "content": """Answer the given question with a single word or phrase""",},
        {
            "role": "user",
            "content": """Question: {question}""",
        },         
    ]

    PROMPT_TEMPLATE_RAG = tokenizer.apply_chat_template(
        prompt_in_chat_format_rag, tokenize = False, add_generation_prompt=True)

    PROMPT_TEMPLATE_LLM = tokenizer.apply_chat_template(
        prompt_in_chat_format_llm, tokenize = False, add_generation_prompt=True)
        
    return PROMPT_TEMPLATE_LLM, PROMPT_TEMPLATE_RAG

def search_context(query, top_k):
    channel = grpc.insecure_channel('localhost:5050')
    stub = faiss_pb2_grpc.FaissStub(channel)
    response = stub.Search(faiss_pb2.SearchRequest(query = query, top_k = top_k))
    return response.contexts, response.scores
    
def generate_query_context_list(q_ls, prompt, is_rag, top_k = 0):
    
    if is_rag:
        for query in q_ls:
            contexts, _ = search_context(query, top_k)
            context = "".join([f" Context {str(i)}: \n" + doc + "\n" for i, doc in enumerate(contexts)])
            yield prompt.format(question = query, context = context)
    else:
        for query in q_ls:
            yield prompt.format(question=query)

def generate_query_context(query, prompt, is_rag, top_k = 0):
    
    if is_rag:
        contexts, _ = search_context(query, top_k)
        context = "".join([f" Context {str(i)}: \n" + doc + "\n" for i, doc in enumerate(contexts)])
        return prompt.format(question = query, context = context)
    else:
        return prompt.format(question=query)
            

def kv_to_np(cache):
    
    res_ls = []
    for layer in cache:
        res_layer = np.array([layer[0].cpu().numpy(), layer[1].cpu().numpy()])
        res_ls.append(res_layer)
    return np.array(res_ls)

def np_to_kv(cache):
    res = tuple(tuple(torch.tensor(kv, dtype=torch.float16).to("cuda") for kv in layer) for layer in cache)
    
    return res

def eval_EM(answer, gts):
    answer = answer.lower()
    gts = [sentence.lower() for sentence in gts]
    
    for gt in gts:
        if gt in answer:
            return True
    return False

def compute_rouge_l(candidate, reference):
    # Create a scorer object with rouge types
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference, candidate)
    
    # Extract and return the ROUGE-L score
    rouge_l_score = scores['rougeL']
    return rouge_l_score.fmeasure

def eval_rougu_L(answer, gts):
    answer = answer.lower()
    gts = [sentence.lower() for sentence in gts]
    
    max_score = 0.0
    
    for gt in gts:
        score = float(compute_rouge_l(answer, gt))
        if  score> max_score:
            max_score = score
    return max_score
    

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def concat_attention(attention):
    num_layers = len(attention[0])
    
    num_output_tokens = len(attention)
    num_entire_tokens = len(attention[-1][0][0][0][0])
    num_heads = len(attention[0][0][0])
    
    layer_ls = []
    
    for i in range(num_layers):
        
        head_ls = []
        
        for j in range(num_heads):
            new_ls = np.zeros((num_entire_tokens, num_entire_tokens))
            
            for k in range(num_output_tokens):
                tmp_ls = np.array(copy.deepcopy(attention[k][i][0][j].cpu()))
                if k == 0:
                    new_ls[:len(tmp_ls),:len(tmp_ls)] = tmp_ls
                else:
                    new_ls[len(tmp_ls[0])-1,:len(tmp_ls[0])] = tmp_ls
            
            head_ls.append(new_ls)
        
        layer_ls.append(np.array(head_ls))    
    
    return np.array(layer_ls)

def concat_attention_no_head(attention):
    num_layers = len(attention[0])
    
    num_output_tokens = len(attention)
    num_entire_tokens = len(attention[-1][0][0][0][0])
    num_heads = len(attention[0][0][0])
    
    
    layer_ls = []
    
    for i in range(num_layers):
        
        head_ls = []
        
        for j in range(num_heads):
            new_ls = np.zeros((num_entire_tokens, num_entire_tokens))
            
            for k in range(num_output_tokens):
                tmp_ls = np.array(copy.deepcopy(attention[k][i][0][j].cpu()))
                if k == 0:
                    new_ls[:len(tmp_ls),:len(tmp_ls)] = tmp_ls
                else:
                    new_ls[len(tmp_ls[0])-1,:len(tmp_ls[0])] = tmp_ls
            
            head_ls.append(new_ls)
        
        layer_ls.append(np.max(np.array(head_ls), axis=0))
    
    return np.array(layer_ls)


def draw_all_attention(array, str_ls, M):
    for n_layer in range(len(array)):
        for n_head in range(len(array[0])):
            draw_attention(array[n_layer,n_head], f"attentions/attention_layer{n_layer}_head{n_head}.jpg", str_ls, M)

def draw_all_attention_no_head(array, str_ls, M):
    for n_layer in range(len(array)):
        draw_attention(array[n_layer], f"attentions/attention_layer{n_layer}.jpg", str_ls, M)
                        
    

def draw_attention(array, name, str_ls, M):
    plt.figure(figsize=(12,10))
    
    #new_array = compress_matrix(array, M)
    new_array = array
    
    plt.imshow(new_array, cmap='viridis', interpolation='nearest')
    plt.colorbar()  
    
    plt.xticks(np.arange(len(new_array)), labels = str_ls, fontsize=10, rotation=90)
    plt.yticks(np.arange(len(new_array)), labels = str_ls, fontsize=10)
    plt.show()
    plt.savefig(name, dpi=300)
    plt.close()
    
def opt_log(opt, log_dir):
    opt_dict = vars(opt)
    with open(f"{log_dir}/opt.yaml", "w") as yml:
        yaml.dump(opt_dict, yml)
        
def compress_matrix(matrix, M):
    N = matrix.shape[0]
    
    dim = N//M if N%M == 0 else N//M + 1
    
    compressed_matrix = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            compressed_matrix[i, j] = np.mean(matrix[i*M:min((i+1)*M, N), j*M:min((j+1)*M, N)])

    
    return compressed_matrix


def cal_score(tr_score_ls, seq_score_ls):
    if len(tr_score_ls) == 0:
        return 0, 0, 0, 0, 0
    assert(len(tr_score_ls) == len(seq_score_ls))

    score_ls = np.exp(tr_score_ls)
    L = len(score_ls)
    
    max_seq_prob = 1 - np.prod(score_ls)
    
    length_normalized_log_prob = np.exp(-1/L*np.sum(np.log(score_ls)))
    
    max_token_prob = 1 - np.min(score_ls)
    
    entropy_ls = []
    
    for score_ls in seq_score_ls:
        tmp = np.array([-prob * np.log(prob) for prob in score_ls if prob != 0])
        entropy_ls.append(tmp.sum())

    mean_token_entropy = sum(entropy_ls) / L
    max_token_entropy = max(entropy_ls)
    
    return max_seq_prob, length_normalized_log_prob, max_token_prob, mean_token_entropy, max_token_entropy

def auroc(df):
    Correct = df['Correct'].tolist()
    Correction = [1 if ind == True else 0 for ind in Correct]
    Sequence_uncertainty = df['Sequence uncertainty'].tolist()
    Perplexity = df['Perplexity'].tolist()
    max_token_uncertainty = df['Max token uncertainty'].tolist()
    Mean_entropy = df['Mean entropy'].tolist()
    Max_token_entropy = df['Max token entropy'].tolist()

    def draw_roc(correction, uncertainty, img_n=None):

        score = [-x for x in uncertainty]

        fpr, tpr, _ = roc_curve(correction, score)
        roc_auc = auc(fpr, tpr)

        if img_n != None:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.legend(loc='lower right')
            plt.savefig(f"{img_n}.png")
        
        return roc_auc

    seq_unc = draw_roc(Correction, Sequence_uncertainty)
    perplexity = draw_roc(Correction, Perplexity)
    max_unc = draw_roc(Correction, max_token_uncertainty)
    mean_ent = draw_roc(Correction, Mean_entropy)
    max_ent = draw_roc(Correction, Max_token_entropy)

    return seq_unc, perplexity, max_unc, mean_ent, max_ent

def prepare_bart():
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def eval_bartscore(ans, gts, bart_scorer):
    max_bart_score = -999.0
    
    for gt in gts:
        bart_s = bart_scorer.score([ans], [gt])[0]
        
        if bart_s > max_bart_score:
            max_bart_score = bart_s
        
    return max_bart_score

    