import argparse

def parse_argument():

    parser = argparse.ArgumentParser()
        
    parser.add_argument("-embed_n",  required=True, choices=["bge", 'miniLM', 'gist'])
    parser.add_argument("-dataset_n", required=True,  choices=["dbricks", 'wiki_simple', 'wiki'])
    parser.add_argument("-chunk_len", required=True,  type=int)
    parser.add_argument("-index_n",  required=True, choices=["FlatIP", 'HNSW'],)
    parser.add_argument("-llm_n",  required=True, choices=["t5", 'zephyr', 'llama2_7b', 'llama2_13b', 'openorca', 'mistral'])
    parser.add_argument("-top_k", required=True, type=int ) 
    parser.add_argument("-num_q", required=True, type=int)
    parser.add_argument("-rerank", action="store_true")
    parser.add_argument("-random", action="store_true")
    parser.add_argument("-no_rag", action="store_true")
    parser.add_argument("-use_input", action="store_true")
        

    return parser.parse_args()

