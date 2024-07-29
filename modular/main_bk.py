from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.schema.document import Document
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForCausalLM, BitsAndBytesConfig, T5Tokenizer, T5ForConditionalGeneration, T5ForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import os
import json
from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision,
)
from tqdm import tqdm

import time
import pandas as pd


def keystoint(x):
    return {int(k): v for k, v in x.items()}


def load_dbricks():

    docs = load_dataset("databricks/databricks-dolly-15k")
    return docs

def load_tqa():
    docs = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    question = docs['question']
    answer = [ans['aliases'] + ans['normalized_aliases'] for ans in docs['answer']]
    return question, answer
    
    
def load_wiki_simple(max_seq_len):
    docs = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)['train']['text']
    docs = [Document(page_content = x) for x in docs]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_seq_len, chunk_overlap=max_seq_len//10)
    chunked_docs = text_splitter.split_documents(docs)
    docs = [doc.page_content for doc in chunked_docs]
    
    return docs


def load_vector_db(index_name):
        
    index = faiss.read_index(index_name)
    with open(f"{index_name}.json", "r") as json_file:
        str_dict = json.load(json_file, object_hook=keystoint)    
    
    return index, str_dict

def generate_vector_db(dataset_n, embedding_model, index_n, index_name, max_seq_len):
    
    match dataset_n:
        case "wiki_simple":
            docs = load_wiki_simple(max_seq_len)
    
        case "dbricks":
            docs = load_dbricks()
    
    vector_ds = np.array(embedding_model.embed_documents(docs))
    str_dict = {index: value for index, value in enumerate(docs)}
    dim = len(vector_ds[0])
    
    match index_n:
        case "FlatIP":
            index = faiss.IndexFlatIP(dim)
        case "HNSW":
            index = faiss.IndexHNSWFlat(dim, 20)
        case "IVFPQ":
            index = faiss.IIndexIVFPQ(dim, 20)
    
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(vector_ds, np.array([ind for ind in range(len(vector_ds))]))
    
    faiss.write_index(index, index_name)
    with open(f"{index_name}.json", "w") as json_file:
            json.dump(str_dict, json_file, indent=None)    
    
    return index, str_dict

def vector_db(dataset_n, index_n, embed_n, embedding_model, max_seq_len):
    
    index_name = f"{dataset_n}_{index_n}_{embed_n}"

    if os.path.exists(index_name):
        index, str_dict = load_vector_db(index_name)
    else:
        index, str_dict =  generate_vector_db(dataset_n, embedding_model, index_n, index_name, max_seq_len)
        
    return index, str_dict

def load_zephyr():
    
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    clean_up_tokenization_spaces=True,
    pad_token_id = tokenizer.eos_token_id
    )    
    

    return READER_LLM, tokenizer

def load_t5():
    
    model_name = "google/flan-t5-large"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text2text-generation",
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1,
    max_new_tokens=1024,
    pad_token_id = tokenizer.eos_token_id
    )    
    

    return READER_LLM, tokenizer

def load_llama2_7b():
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    clean_up_tokenization_spaces=True,
    pad_token_id = tokenizer.eos_token_id
    )    
    
    

    return READER_LLM, tokenizer

def load_llama2_13b():
    
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    clean_up_tokenization_spaces=True,
    pad_token_id = tokenizer.eos_token_id
    )    

    return READER_LLM, tokenizer


def make_llm_pipeline(model_name):

    match model_name:
        case "zephyr":
            READER_LLM, tokenizer = load_zephyr()
        case "t5":
            READER_LLM, tokenizer = load_t5()
        case "llama2_7b":
            READER_LLM, tokenizer = load_llama2_7b()
        case "llama2_13b":
            READER_LLM, tokenizer = load_llama2_13b()

   
    
    prompt_in_chat_format_rag = [
        {
            "role": "system",
            "content": """
    Give an answer to the question based on the context.
""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Question: {question}""",
        },         
    ]
    
    prompt_in_chat_format_llm = [
        {
            "role": "system",
            "content": """
    Give an answer to the question.""",},
        {
            "role": "user",
            "content": """
    Question: {question}""",
        },         
    ]

    RAG_PROMPT_TEMPLATE_RAG = tokenizer.apply_chat_template(
        prompt_in_chat_format_rag, tokenize=False, add_generation_prompt=True
    )
  
    RAG_PROMPT_TEMPLATE_LLM = tokenizer.apply_chat_template(
        prompt_in_chat_format_llm, tokenize=False, add_generation_prompt=True
    )

    return RAG_PROMPT_TEMPLATE_LLM, RAG_PROMPT_TEMPLATE_RAG, READER_LLM



def rerank(query, context_ls, tokenizer, reranker, top_k):

    reranker.eval()

    pairs = [[ct, query] for ct in context_ls]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
    _, top_indices = torch.topk(scores, top_k)
    new_context_ls = [ct for ind, ct in enumerate(context_ls) if ind in top_indices]

    return new_context_ls
        
        

def question_and_test_input(EMBEDDING_MODEL_NAME, DATASET_NAME, INDEX_NAME, LLM_NAME, TOP_K):
    
    match EMBEDDING_MODEL_NAME:
        case "miniLM":
            embed_n = "../all-MiniLM-L6-v2"
        case "bge":
            embed_n = "BAAI/bge-base-en-v1.5"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_n,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )
    
    max_seq_len = SentenceTransformer(embed_n).max_seq_length
    
    index, str_dict = vector_db(DATASET_NAME, INDEX_NAME, EMBEDDING_MODEL_NAME, embedding_model, max_seq_len)
    prompt_template_llm, prompt_template_rag, llm_pipeline = make_llm_pipeline(LLM_NAME)
    
    log_n = f"log_{DATASET_NAME}_{EMBEDDING_MODEL_NAME}_{INDEX_NAME}_{LLM_NAME}.csv"
    
    if os.path.exists(log_n):
        df = pd.read_csv(log_n)
        df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    else:
        df = pd.DataFrame(columns = ['question', 'context', 'LLM_answer', "RAG_answer", "Time_LLM", "Time_search", "Time_RAG"])
    
    tokenizer_rerank = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
    reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
    reranker.eval()    

    while True:   
        question = input("Type Question: ")
        
        if question == "exit":
            df.to_csv(log_n, index=True)
            exit()
        
        query_vector = np.array(embedding_model.embed_query(question)).reshape(1, -1)
    
        print("-------------------------------------LLM----------------------------------")
        

        LLM_prompt = prompt_template_llm.format(question = question)
       
        start_llm = time.perf_counter()
              
        llm_ans = llm_pipeline(LLM_prompt)[0]["generated_text"]
        
        t_llm = time.perf_counter()-start_llm
        
        print(llm_ans)
        
        print('---------------------------------------------------------------------------')
        print('--------------------------------------RAG----------------------------------')
        start_ind = time.perf_counter()  
        _, indices = index.search(query_vector, TOP_K*5)
        t_ind = time.perf_counter()-start_ind
        
        retrieved_docs = [str_dict[indices[0][i]] for i in range(TOP_K*5)]

        retrieved_docs = rerank(question, retrieved_docs, tokenizer_rerank, reranker, TOP_K)

        context = "".join([f"Doc 1: " + doc + "\n" for i, doc in enumerate(retrieved_docs)])
    
        final_prompt = prompt_template_rag.format(question = question, context = context)
        start_rag = time.perf_counter()  
        rag_ans = llm_pipeline(final_prompt)[0]["generated_text"]
        t_rag = time.perf_counter()-start_rag
        print(rag_ans)
        
        print('---------------------------------------------------------------------------')
       
        df.loc[len(df)] = [question, context, llm_ans, rag_ans, t_llm, t_ind, t_rag]


def check_ans(answer, gts):
    for gt in gts:
        if gt in answer:
            return True
    return False

def question_and_test_tqa(EMBEDDING_MODEL_NAME, DATASET_NAME, INDEX_NAME, LLM_NAME, TOP_K, num_q):
    
    match EMBEDDING_MODEL_NAME:
        case "miniLM":
            embed_n = "../all-MiniLM-L6-v2"
        case "bge":
            embed_n = "BAAI/bge-base-en-v1.5"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_n,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )
    
    max_seq_len = SentenceTransformer(embed_n).max_seq_length
    
    index, str_dict = vector_db(DATASET_NAME, INDEX_NAME, EMBEDDING_MODEL_NAME, embedding_model, max_seq_len)
    prompt_template_llm, prompt_template_rag, llm_pipeline = make_llm_pipeline(LLM_NAME)
    
    log_n = f"log_{DATASET_NAME}_{EMBEDDING_MODEL_NAME}_{INDEX_NAME}_{LLM_NAME}_TQA.csv"
    
    if os.path.exists(log_n):
        df = pd.read_csv(log_n)
        df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    else:
        df = pd.DataFrame(columns = ['question', 'GT', 'context', 'LLM_answer', "RAG_answer", "Time_LLM", "Time_search", "Time_RAG", "LLM_correct", "RAG_correct"])
    
    tokenizer_rerank = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
    reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
    reranker.eval()    

    question_ls, answer_ls = load_tqa()

    for ind in tqdm(range(num_q)):   

        question = question_ls[ind]
        answer = answer_ls[ind]
        
        query_vector = np.array(embedding_model.embed_query(question)).reshape(1, -1)
    
        LLM_prompt = prompt_template_llm.format(question = question)
       
        start_llm = time.perf_counter()
              
        llm_ans = llm_pipeline(LLM_prompt)[0]["generated_text"]
        
        t_llm = time.perf_counter()-start_llm
        
        llm_correct = check_ans(llm_ans, answer)
        

        start_ind = time.perf_counter()  
        _, indices = index.search(query_vector, TOP_K*5)
        t_ind = time.perf_counter()-start_ind
        
        retrieved_docs = [str_dict[indices[0][i]] for i in range(TOP_K*5)]

        retrieved_docs = rerank(question, retrieved_docs, tokenizer_rerank, reranker, TOP_K)

        context = "".join([f" Document {str(i)}:::\n" + doc + "\n" for i, doc in enumerate(retrieved_docs)])
    
        final_prompt = prompt_template_rag.format(question = question, context = context)
        start_rag = time.perf_counter()  
        rag_ans = llm_pipeline(final_prompt)[0]["generated_text"]
        t_rag = time.perf_counter()-start_rag
        rag_correct = check_ans(rag_ans, answer)
        
        df.loc[len(df)] = [question, answer, context, llm_ans, rag_ans, t_llm, t_ind, t_rag, llm_correct, rag_correct]
   
   
    df.to_csv(log_n, index=True)   

if __name__ == '__main__':
    
    EMBEDDING_MODEL_NAME = "bge"
    DATASET_NAME = "wiki_simple"
    INDEX_NAME = "FlatIP"
    LLM_NAME = "llama2_7b"
    Top_k = 5
    NUM_QUESTION = 20
    
    #question_and_test_input(EMBEDDING_MODEL_NAME, DATASET_NAME, INDEX_NAME, LLM_NAME, Top_k)    
    question_and_test_tqa(EMBEDDING_MODEL_NAME, DATASET_NAME, INDEX_NAME, LLM_NAME, Top_k, NUM_QUESTION)    



    
    
    
    

