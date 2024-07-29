import argparse

def parse_argument():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-label", required=True, choices = ['correct', 'Sequence uncertainty', 'Perplexity', 'Max token uncertainty', 'Mean entropy', 'Max token entropy', 'twin_2', 'twin_4', 'relative uncertainty', 'relative uncertainty 1', 'relative uncertainty 2'])
    
    parser.add_argument("-file_n", required=True, type=str)
    parser.add_argument("-file_2_n", type=str)
    
    
    parser.add_argument("-model_path", type=str)
    
    parser.add_argument("-hybrid", type=int, choices=[0, 1, 2], default=0)
                         

    return parser.parse_args()

