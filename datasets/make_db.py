import faiss, json, os
import torch
from transformers import AutoTokenizer, AutoModel


def keystoint(x):
    return {int(k): v for k, v in x.items()}

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')
    
    dict_path = "wiki_256.json"
    
    with open(dict_path, "r") as json_file:
        str_dict = json.load(json_file, object_hook=keystoint)    

    