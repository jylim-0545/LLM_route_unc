from concurrent import futures
import grpc
import faiss_pb2
import faiss_pb2_grpc
import faiss
import json
from langchain_community.embeddings import HuggingFaceEmbeddings,  HuggingFaceBgeEmbeddings
import numpy as np

def keystoint(x):
    return {int(k): v for k, v in x.items()}

def load_embed(embed_n):
    match embed_n:
        case "miniLM":
            embed_n = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_model = HuggingFaceEmbeddings(
            model_name=embed_n,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
            )
        case "bge":
            embed_n = "BAAI/bge-base-en-v1.5"
            embedding_model = HuggingFaceBgeEmbeddings(
            model_name=embed_n,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
            query_instruction = "Represent this sentence for searching relevant passages:"
            )
        case "gist":
            embed_n = "avsolatorio/GIST-Embedding-v0" 
            embedding_model = HuggingFaceEmbeddings(
            model_name=embed_n,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
            )        
    return embedding_model    
                



class FaissService(faiss_pb2_grpc.FaissServicer):
    def __init__(self, ds_n, chunk_len, index_n, embed_n):
        
        self.ds_n, self.chunk_len, self.index_n = ds_n, chunk_len, index_n
        
        index_name = f"../datasets/{ds_n}_{chunk_len}_{index_n}_{embed_n}"
        dict_name = f"../datasets/{ds_n}_{chunk_len}"
        
        
        
        self.index = faiss.read_index(index_name)
        with open(f"{dict_name}.json", "r") as json_file:
            self.str_dict = json.load(json_file, object_hook=keystoint)    
        
        self.embed = load_embed(embed_n)
        
        print("Server is ready")
        print(len(self.str_dict))
        print(f"Dataset: {ds_n}_{chunk_len} | Index: {index_n} | Embedding model: {embed_n}")
         
    def Search(self, request, context):

        query_vector = np.array(self.embed.embed_query(request.query)).reshape(1, -1)
        top_k = request.top_k
        
        scores, indices = self.index.search(query_vector, top_k)
        retrieved_docs = [self.str_dict[indices[0][i]] for i in range(top_k)]
        
        return faiss_pb2.SearchResponse(contexts = retrieved_docs, scores = scores[0])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    
    FS = FaissService("wiki", 256, "HNSW", "gist")
    
    faiss_pb2_grpc.add_FaissServicer_to_server(FS, server)
    server.add_insecure_port('localhost:5050')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
