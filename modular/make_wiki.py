from datasets import load_dataset, Dataset, DownloadConfig
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
import faiss, json, os
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool
import math
'''
docs = load_dataset("wikipedia", "20220301.en", download_config=DownloadConfig(resume_download=True))['train']['text']

text_splitter = SentenceSplitter(chunk_size = 128, paragraph_separator='\n\n', chunk_overlap=0)

def process_document(doc):
    tmp = text_splitter.split_text(doc)
    result = [doc_s for doc_s in tmp if len(doc_s) > 30]
    return result

with Pool() as pool:
    results = list(tqdm(pool.imap(process_document, docs), total=len(docs)))

doc_dict = {}
for result in tqdm(results):
    for doc_s in result:
        doc_dict[len(doc_dict)] = doc_s

with open(f"wiki_tmp.json", "w") as json_file:
    json.dump(doc_dict, json_file, indent=None)  
'''
if __name__ == '__main__':


    docs = load_dataset("wikipedia", "20220301.en", download_config=DownloadConfig(resume_download=True))['train']['text']

    text_splitter = SentenceSplitter(chunk_size = 256, chunk_overlap=0)

    def process_document(doc):
        tmp = text_splitter.split_text(doc)
        result = [doc_s for doc_s in tmp if len(doc_s) > 30]
        return result

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_document, docs), total=len(docs)))

    doc_dict = {}
    for result in tqdm(results):
        for doc_s in result:
            doc_dict[len(doc_dict)] = doc_s

    with open(f"wiki_256.json", "w") as json_file:
        json.dump(doc_dict, json_file, indent=None)  


    embed_n = "avsolatorio/GIST-Embedding-v0" 
    dict_path = "../datasets/wiki_256.json"
    
    
    
    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open(dict_path, "r") as json_file:
        str_dict = json.load(json_file, object_hook=keystoint)    

    docs = list(str_dict.values())



    embedding_model = HuggingFaceEmbeddings(
    model_name=embed_n,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    show_progress = True
    )
    
    num_partitions = 20
    partition_size = len(docs) // num_partitions
    
    for i in range(num_partitions):
        start_ind = i * partition_size
        end_ind = start_ind + partition_size if i < num_partitions - 1 else len(docs)
        partition_docs = docs[start_ind:end_ind]
        embed_ls = embedding_model.embed_documents(partition_docs)
        np.save(f'wiki_gist/partition_{i+1}.npy', np.array(embed_ls, dtype=np.float32))
    
    all_data = []
    
    for i in tqdm(range(20)):    
        npy = np.load(f'wiki_gist/partition_{i+1}.npy')
        all_data.append(npy)
    
    embed_np = np.concatenate(all_data, axis=0)
    

    dim = len(embed_np[0])
    index = faiss.IndexHNSWFlat(dim, 20, faiss.METRIC_INNER_PRODUCT)
    #index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, nbits)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(embed_np, np.arange(embed_np.shape[0]))
    faiss.write_index(index, "wiki_256_HNSW_gist")
        