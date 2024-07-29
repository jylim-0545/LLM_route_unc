import argparse

def parse_argument():

    parser = argparse.ArgumentParser()
        
    parser.add_argument("-llm_n",  required=True, choices=["t5", 'zephyr', 'llama2_7b', 'llama2_13b', 'openorca', 'mistral', 'llama3_8b', 'opt_125m', 'opt_350m', 'opt_1.3b', 'opt_2.7b', 'opt_6.7b', 'opt_13b', 'opt_30b'])
    parser.add_argument("-top_k", required=True, type=int ) 
    parser.add_argument("-num_q", required=False, type=int)
    parser.add_argument("-rerank", action="store_true")
    parser.add_argument("-random", action="store_true")
    parser.add_argument("-is_rag", action="store_true")
    parser.add_argument("-use_input", action="store_true")
    parser.add_argument("-q_number", type=str)
    parser.add_argument("-ds", type=str)
                         

    return parser.parse_args()

