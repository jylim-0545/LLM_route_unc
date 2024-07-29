
import dataset
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter
import faiss
import numpy as np
from tqdm import tqdm
EMBEDDING_NAME = 'gist'
ds_n = 'wiki'
index_n = "HNSW"
index_name = 'tmp'


'''embedding_model = HuggingFaceBgeEmbeddings(
model_name=embed_n,
model_kwargs={"device": "cuda"},
encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
query_instruction = "Represent this sentence for searching relevant passages:"
)'''


index_name = f"datasets/{ds_n}_256_{index_n}_{EMBEDDING_NAME}"
index = faiss.read_index(index_name)    
print(index.is_trained)

nb = 1
d = 768

xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

for _ in tqdm(range(1000)):
    index.add_with_ids(xb, 1)








