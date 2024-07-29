from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig,  AutoModelForSeq2SeqLM

import torch
import util

model_ls = {'zephyr': "HuggingFaceH4/zephyr-7b-beta", 'llama2_7b': "meta-llama/Llama-2-7b-hf", 'llama2_13b': "meta-llama/Llama-2-13b-hf", 'openorca': "Open-Orca/Mistral-7B-OpenOrca", 'mistral':"mistralai/Mistral-7B-Instruct-v0.2", 'llama3_8b': "meta-llama/Meta-Llama-3-8B", 'opt_125m': "facebook/opt-125m", 'opt_350m': "facebook/opt-350m", 'opt_1.3b': "facebook/opt-1.3b", 'opt_2.7b': "facebook/opt-2.7b", 'opt_6.7b': "facebook/opt-6.7b", 'opt_13b': "facebook/opt-13b", 'opt_30b': "facebook/opt-30b"}

def load_model(model_n):
    
    model_name = model_ls[model_n]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)    
    READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    do_sample=False,
    temperature=None,
    return_full_text = False,
    repetition_penalty=1.0,
    max_new_tokens=100,
    clean_up_tokenization_spaces=True,
    pad_token_id = tokenizer.eos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    output_scores=True,
    output_logits=True
    )    
    
    if model_n == "llama3_8b":
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"

    llm_prompt, rag_prompt = util.prompt_template(model_n, tokenizer)

    return READER_LLM, tokenizer, model, llm_prompt, rag_prompt
       