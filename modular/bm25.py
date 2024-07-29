from rank_bm25 import BM25Okapi
import pickle
from datasets import load_dataset

'''




tokenized_corpus = [tokenizer(doc) for doc in docs]

bm25 = BM25Okapi(tokenized_corpus)

with open('bm25_temp', 'wb') as bm25result_file:
    pickle.dump(bm25, bm25result_file)
'''
def tokenizer(sent):
    return sent.split(" ")
docs = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)['train']['text']

with open('bm25_temp', 'rb') as bm25result_file:
    bm25 = pickle.load(bm25result_file)

query = "Who is the current president of the USA ?"
token_query = tokenizer(query)

print(bm25.get_top_n(token_query, docs, n=10))