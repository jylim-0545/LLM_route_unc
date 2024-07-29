from datasets import load_dataset, Dataset, DownloadConfig
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llama_index.core.node_parser import SentenceSplitter

import faiss, json, os
import numpy as np
import random
from tqdm import tqdm

def keystoint(x):
    return {int(k): v for k, v in x.items()}

def load_dbricks():

    docs = load_dataset("databricks/databricks-dolly-15k")['train']['context']
    docs = [Document(page_content = x) for x in docs if len(x) > 3]
    
    return docs


def load_wiki_simple():
    
    docs = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)['train']['text']
    #docs = [Document(page_content = x) for x in docs]
    return docs

def load_wiki():
    
    docs = load_dataset("wikipedia", "20220301.en", download_config=DownloadConfig(resume_download=True))['train']['text']
    #docs = [Document(page_content = x) for x in docs]
    
    return docs



def load_vector_db(index_name, dict_name):
    print("Load DB!")
    index = faiss.read_index(index_name)
    with open(f"{dict_name}.json", "r") as json_file:
        str_dict = json.load(json_file, object_hook=keystoint)    
    
    return index, str_dict

def generate_vector_db(dataset_n, chunk_len, embedding_model, index_n, index_name, dict_name):


    if os.path.exists(f"{dict_name}.json"):
        print("Load dict!")
        with open(f"{dict_name}.json", "r") as json_file:
            str_dict = json.load(json_file, object_hook=keystoint)
            docs = list(str_dict.values())
    else:
        match dataset_n:
            case "wiki_simple":
                docs = load_wiki_simple()
            case "dbricks":
                docs = load_dbricks()
            case "wiki":
                docs = load_wiki()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_seq_len, chunk_overlap=max_seq_len//10)
        text_splitter = SentenceSplitter(chunk_size = chunk_len, paragraph_separator='\n\n', chunk_overlap=0)

        docs = [doc for doc in text_splitter.split_texts(docs) if len(doc) > 30]
        
        str_dict = {index: value for index, value in enumerate(docs)}
        with open(f"{dict_name}.json", "w") as json_file:
                json.dump(str_dict, json_file, indent=None)    




    vector_ds = np.array(embedding_model.embed_documents(docs), dtype = np.float32)
    

    dim = len(vector_ds[0])
    
    match index_n:
        case "FlatIP":
            index = faiss.IndexFlatIP(dim)
        case "HNSW":
            index = faiss.IndexHNSWFlat(dim, 20, faiss.METRIC_INNER_PRODUCT)
        case "IVFPQ":
            index = faiss.IndexIVFPQ(quantizer, dim, 20)
    
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(vector_ds, np.array([ind for ind in range(len(vector_ds))]))
    
    faiss.write_index(index, index_name)
    

    
    return index, str_dict

def vector_db(dataset_n, chunk_len, index_n, index_name, embedding_model, dict_name):
    
    if os.path.exists(index_name):
        index, str_dict = load_vector_db(index_name, dict_name)
    else:
        index, str_dict =  generate_vector_db(dataset_n, chunk_len, embedding_model, index_n, index_name, dict_name)
        
    return index, str_dict

def load_tqa(num_q, is_random=False):
    docs = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    question = docs['question']
    answer = [ans['aliases'] + ans['normalized_aliases'] for ans in docs['answer']]
    
    if is_random:
        tmp = list(zip(question, answer))
        random.shuffle(tmp)
        question, answer = zip(*tmp)
    
    return question[:num_q], answer[:num_q]
    


