from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig,  AutoModelForSeq2SeqLM

import torch

model_ls = {'zephyr': "HuggingFaceH4/zephyr-7b-beta", "t5": "google/flan-t5-large", 'llama2_7b': "meta-llama/Llama-2-7b-chat-hf", 'llama2_13b': "meta-llama/Llama-2-13b-chat-hf", 'openorca': "Open-Orca/Mistral-7B-OpenOrca", 'mistral':"mistralai/Mistral-7B-Instruct-v0.2"}

def load_model(model_n):
    
    model_name = model_ls[model_n]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_n == 't5':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config)
        READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text2text-generation',
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        max_new_tokens=100,
        clean_up_tokenization_spaces=True,
        pad_token_id = tokenizer.eos_token_id
        )      
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")    
        READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        do_sample=True,
        temperature=0.1,
        return_full_text = False,
        repetition_penalty=1.1,
        max_new_tokens=100,
        clean_up_tokenization_spaces=True,
        pad_token_id = tokenizer.eos_token_id
        )    

    
    return READER_LLM, tokenizer
       