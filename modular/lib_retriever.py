
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import dataset
import numpy as np
import torch
from util import prompt_template

class Retriever():
    def __init__(self, EMBEDDING_NAME, ds_n, ds_chunk_len, index_n, tokenizer_llm, model_n, reranker_n=None, no_rag = False):
        
        if not no_rag:
        
            match EMBEDDING_NAME:
                case "miniLM":
                    embed_n = "sentence-transformers/all-MiniLM-L6-v2"
                    self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embed_n,
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
                    )
                case "bge":
                    embed_n = "BAAI/bge-base-en-v1.5"
                    self.embedding_model = HuggingFaceBgeEmbeddings(
                    model_name=embed_n,
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
                    query_instruction = "Represent this sentence for searching relevant passages:"
                    )
                case "gist":
                    embed_n = "avsolatorio/GIST-Embedding-v0" 
                    self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embed_n,
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
                    )            
            

            index_name = f"datasets/{ds_n}_{ds_chunk_len}_{index_n}_{EMBEDDING_NAME}"
            dict_name = f"datasets/{ds_n}_{ds_chunk_len}"
            print(index_name)
        
            self.index, self.str_dict = dataset.vector_db(ds_n, ds_chunk_len, index_n, index_name, self.embedding_model, dict_name)
        
        
        if reranker_n:
            self.tokenizer_rerank = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
            self.reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
            self.reranker.eval()  
        else:
            self.rerank = None
        
        self.prompt_llm, self.prompt_rag= prompt_template(model_n, tokenizer_llm)
        
        self.context_ls = []
        self.generate_latency = 0
        self.retrieve_latency = 0
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.retrieve_time = 0
        self.scores = []
    
    def llm_prompt_seq(self, question_ls):
        for question in question_ls:
            yield self.prompt_llm.format(question = question)
        
    def retrieve(self, TOP_K, question):
        
        self.start.record()
        
        query_vector = np.array(self.embedding_model.embed_query(question)).reshape(1, -1)


        scores, indices = self.index.search(query_vector, TOP_K)
        retrieved_docs = [self.str_dict[indices[0][i]] for i in range(TOP_K)]

        self.end.record()
        self.end.synchronize()
        self.retrieve_time = self.start.elapsed_time(self.end)
        self.scores = scores[0]

        context = "".join([f" Context {str(i)}: \n" + doc + "\n" for i, doc in enumerate(retrieved_docs)])
        
        return context
    
    def retrieve_and_rerank(self, TOP_K, question):
        self.start.record()
        
        query_vector = np.array(self.embedding_model.embed_query(question)).reshape(1, -1)
    
        _, indices = self.index.search(query_vector, TOP_K*TOP_K)
        context_ls = [self.str_dict[indices[0][i]] for i in range(TOP_K*TOP_K)]

        pairs = [[ct, question] for ct in context_ls]
        with torch.no_grad():
            inputs = self.tokenizer_rerank(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.reranker(**inputs, return_dict=True).logits.view(-1, ).float()
        _, top_indices = torch.topk(scores, TOP_K)
        new_context_ls = [ct for ind, ct in enumerate(context_ls) if ind in top_indices]        
        
        self.end.record()
        self.end.synchronize()
        self.retrieve_time = self.start.elapsed_time(self.end)
        self.scores = scores[0]
        
        
        context = "".join([f" Context {str(i)}: \n" + doc + "\n" for i, doc in enumerate(new_context_ls)])
        
        return context
    

    
    
    def retrieve_and_prompt(self, TOP_K, question):
        context = self.retrieve(TOP_K, question)
        self.context_ls.append(context)
        return self.prompt_rag.format(question = question, context = context)
        
    def retrieve_and_rerank_prompt(self, TOP_K, question):
        context = self.retrieve_and_rerank(TOP_K, question)
        self.context_ls.append(context)
        return self.prompt_rag.format(question = question, context = context)
        
    


    
    def rag_prompt_seq(self, question_ls, TOP_K):
        self.context_ls = []
        for question in question_ls:
            yield self.retrieve_and_prompt(TOP_K, question)

    def rag_rerank_prompt_seq(self, question_ls, TOP_K):
        self.context_ls = []
        for question in question_ls:
            yield self.retrieve_and_rerank_prompt(TOP_K, question)




        
        