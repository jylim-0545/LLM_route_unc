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
import time
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

def keystoint(x):
    return {int(k): v for k, v in x.items()}


if __name__ == '__main__':


    n = 2000

    with open(f"datasets/wiki_128.json", "r") as json_file:
        a = json_file.read(n) 
    
    with open(f"datasets/wiki_256.json", "r") as json_file:
        b = json_file.read(n) 
    
    print(a)
    print("\n-------------\n")
    print(b)


